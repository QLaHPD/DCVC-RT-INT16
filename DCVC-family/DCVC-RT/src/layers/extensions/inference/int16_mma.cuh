// Exact INT16 dot products on Ampere integer Tensor Cores.
// An INT16 v = unsigned(low byte) + 256 * signed(high byte).  Four
// non-saturating byte products recover its dot product modulo 2^32; no
// quantization, floating point, or assumptions about the value range are used.
// Fragment mapping: NVIDIA PTX ISA, Matrix Fragments for mma.m16n8k32.

template <bool SIGNED_A, bool SIGNED_B>
__device__ __forceinline__ void mma_bytes(uint32_t* c, const uint32_t* a, const uint32_t* b)
{
#if __CUDA_ARCH__ >= 800
#define DCVC_MMA_BYTES(ATYPE, BTYPE) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32." ATYPE "." BTYPE ".s32 " \
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" \
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) \
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]))
    if constexpr (SIGNED_A && SIGNED_B) { DCVC_MMA_BYTES("s8", "s8"); }
    else if constexpr (SIGNED_A) { DCVC_MMA_BYTES("s8", "u8"); }
    else if constexpr (SIGNED_B) { DCVC_MMA_BYTES("u8", "s8"); }
    else { DCVC_MMA_BYTES("u8", "u8"); }
#undef DCVC_MMA_BYTES
#endif
}

__device__ __forceinline__ void split_four_int16(const int16_t* ptr, uint32_t& lo, uint32_t& hi)
{
    const uint32_t p = reinterpret_cast<const uint32_t*>(ptr)[0];
    const uint32_t q = reinterpret_cast<const uint32_t*>(ptr)[1];
    lo = __byte_perm(p, q, 0x6420);
    hi = __byte_perm(p, q, 0x7531);
}

template <bool WITH_BIAS, bool CONV3, int BN = 32, bool WITH_RESIDUAL = false>
__global__ __launch_bounds__(128) void conv2d_int16_mma_kernel(
    int16_t* __restrict__ out, const int16_t* __restrict__ x,
    const int16_t* __restrict__ weight, const int16_t* __restrict__ bias,
    int M, int N, int K, int H, int W, const int16_t* residual = nullptr)
{
#if __CUDA_ARCH__ >= 800
    constexpr int BM = 64, BK = 32;
    static_assert(BN == 16 || BN == 32 || BN == 64);
    __shared__ __align__(4) int16_t a_tile[BM][BK];
    // Padding reduces bank conflicts during the coalesced NCHW -> column-major store.
    __shared__ __align__(4) int16_t b_tile[BN][BK + 2];
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int group = lane / 4, quad = lane % 4;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    uint32_t ll[BN/8][4] = {}, cross[BN/8][4] = {}, hh[BN/8][4] = {};
    for (int k0 = 0; k0 < K; k0 += BK) {
        #pragma unroll
        for (int i = tid; i < BM * BK; i += 128) {
            const int m = m0 + i / BK, k = k0 + i % BK;
            a_tile[i / BK][i % BK] = (m < M && k < K) ? weight[static_cast<int64_t>(m)*K+k] : 0;
        }
        #pragma unroll
        for (int i = tid; i < BK * BN; i += 128) {
            const int k = k0 + i / BN, n = n0 + i % BN;
            int16_t value = 0;
            if (k < K && n < N) {
                if constexpr (CONV3) {
                    const int y = n / W + (k % 9) / 3 - 1;
                    const int col = n % W + k % 3 - 1;
                    if (y >= 0 && y < H && col >= 0 && col < W)
                        value = x[(static_cast<int64_t>(blockIdx.z)*(K/9)+k/9)*N+y*W+col];
                } else {
                    value = x[(static_cast<int64_t>(blockIdx.z)*K+k)*N+n];
                }
            }
            b_tile[i % BN][i / BN] = value;
        }
        __syncthreads();
        uint32_t al[4], ah[4];
        #pragma unroll
        for (int r = 0; r < 4; ++r)
            split_four_int16(&a_tile[warp*16+group+(r%2)*8][quad*4+(r/2)*16], al[r], ah[r]);
        #pragma unroll
        for (int tile = 0; tile < BN/8; ++tile) {
            uint32_t bl[2], bh[2];
            #pragma unroll
            for (int r = 0; r < 2; ++r)
                split_four_int16(&b_tile[tile*8+group][quad*4+r*16], bl[r], bh[r]);
            mma_bytes<false, false>(ll[tile], al, bl);
            mma_bytes<true, false>(cross[tile], ah, bl);
            mma_bytes<false, true>(cross[tile], al, bh);
            mma_bytes<true, true>(hh[tile], ah, bh);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int tile = 0; tile < BN/8; ++tile) {
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int m = m0 + warp*16 + group + (r/2)*8;
            const int n = n0 + tile*8 + quad*2 + r%2;
            if (m < M && n < N) {
                const uint32_t sum = ll[tile][r] + (cross[tile][r] << 8) + (hh[tile][r] << 16);
                int32_t value = round_divide_int(static_cast<int32_t>(sum), WEIGHT_SCALE);
                if constexpr (WITH_BIAS) value += static_cast<int32_t>(bias[m]);
                const int64_t index = (static_cast<int64_t>(blockIdx.z)*M+m)*N+n;
                // Preserve the convolution's saturation before adding the residual.
                if constexpr (WITH_RESIDUAL)
                    value = static_cast<int32_t>(clip_to_int16(value)) + residual[index];
                out[index] = clip_to_int16(value);
            }
        }
    }
#else
    // A build containing only pre-Ampere PTX can be JIT-loaded on a newer GPU.
    // Keep that configuration correct too: its PTX has no MMA instructions.
    for (int i = threadIdx.x; i < 64*BN; i += blockDim.x) {
        const int m = blockIdx.y*64 + i/BN, n = blockIdx.x*BN + i%BN;
        if (m >= M || n >= N) continue;
        uint32_t sum = 0;
        for (int k = 0; k < K; ++k) {
            int16_t v;
            if constexpr (CONV3) {
                const int y = n/W + (k%9)/3 - 1, col = n%W + k%3 - 1;
                v = (y >= 0 && y < H && col >= 0 && col < W)
                    ? x[(static_cast<int64_t>(blockIdx.z)*(K/9)+k/9)*N+y*W+col] : 0;
            } else {
                v = x[(static_cast<int64_t>(blockIdx.z)*K+k)*N+n];
            }
            sum += static_cast<uint32_t>(static_cast<int32_t>(v)*weight[static_cast<int64_t>(m)*K+k]);
        }
        int32_t value = round_divide_int(static_cast<int32_t>(sum), WEIGHT_SCALE);
        if constexpr (WITH_BIAS) value += static_cast<int32_t>(bias[m]);
        const int64_t index = (static_cast<int64_t>(blockIdx.z)*M+m)*N+n;
        if constexpr (WITH_RESIDUAL)
            value = static_cast<int32_t>(clip_to_int16(value)) + residual[index];
        out[index] = clip_to_int16(value);
    }
#endif
}

bool int16_mma_supported(int device)
{
    // Avoid querying the driver for each convolution; still support switching devices.
    static thread_local int cached_device = -1;
    static thread_local bool supported = false;
    if (cached_device != device) {
        int major = 0;
        const auto status = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
        supported = status == cudaSuccess && major >= 8;
        cached_device = device;
    }
    return supported;
}
