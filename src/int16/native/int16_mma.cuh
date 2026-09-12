// Derived from the managed RT integer MMA fragment layout (MIT-licensed).
// https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma
// v = unsigned(lo8) + 256*signed(hi8). For K <= 32768, each of LL, cross,
// and HH fits int32: |cross| <= 65280*K < 2^31, LL <= 65025*K < 2^31.
// Recombination is INT64, unlike RT's modulo-2^32 recombination.

template<bool SA, bool SB>
__device__ __forceinline__ void uf_mma_bytes(uint32_t* c,const uint32_t* a,const uint32_t* b) {
#if __CUDA_ARCH__ >= 800
#define UF_MMA(A,B) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32." A "." B ".s32 " \
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" \
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) \
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]))
    if constexpr(SA && SB) { UF_MMA("s8","s8"); }
    else if constexpr(SA) { UF_MMA("s8","u8"); }
    else if constexpr(SB) { UF_MMA("u8","s8"); }
    else { UF_MMA("u8","u8"); }
#undef UF_MMA
#endif
}

__device__ __forceinline__ void uf_split(const int16_t* p,uint32_t& lo,uint32_t& hi) {
    const uint32_t a=reinterpret_cast<const uint32_t*>(p)[0];
    const uint32_t b=reinterpret_cast<const uint32_t*>(p)[1];
    lo=__byte_perm(a,b,0x6420); hi=__byte_perm(a,b,0x7531);
}

template<bool POINTWISE>
__device__ int16_t uf_mma_input(const int16_t* x,int batch,int k,int n,int K,
    int H,int W,int OW,int KH,int KW,int SH,int SW,int PH,int PW) {
    if constexpr(POINTWISE) return x[(static_cast<int64_t>(batch)*K+k)*H*W+n];
    const int area=KH*KW;
    const int y=(n/OW)*SH+(k%area)/KW-PH;
    const int col=(n%OW)*SW+k%KW-PW;
    if(y<0 || y>=H || col<0 || col>=W) return 0;
    return x[((static_cast<int64_t>(batch)*(K/area)+k/area)*H+y)*W+col];
}

template<bool POINTWISE, int A_PADDING=0, int TILE_N=32, int TILE_K=32>
__global__ __launch_bounds__(128) void uf_mma_conv(int16_t* out,const int16_t* x,
    const int16_t* weight,const int16_t* bias,int M,int N,int K,int H,int W,int OW,
    int KH,int KW,int SH,int SW,int PH,int PW) {
    constexpr int BM=64, BN=TILE_N, BK=TILE_K;
#if __CUDA_ARCH__ >= 800
    __shared__ __align__(4) int16_t a_tile[BM][BK+A_PADDING];
    __shared__ __align__(4) int16_t b_tile[BN][BK+2];
    const int tid=threadIdx.x,warp=tid/32,lane=tid%32;
    const int group=lane/4,quad=lane%4,m0=blockIdx.y*BM,n0=blockIdx.x*BN;
    uint32_t ll[BN/8][4]={},cross[BN/8][4]={},hh[BN/8][4]={};
    for(int k0=0;k0<K;k0+=BK) {
        #pragma unroll
        for(int i=tid;i<BM*BK;i+=128) {
            const int m=m0+i/BK,k=k0+i%BK;
            a_tile[i/BK][i%BK]=(m<M && k<K)?weight[static_cast<int64_t>(m)*K+k]:0;
        }
        #pragma unroll
        for(int i=tid;i<BK*BN;i+=128) {
            const int k=k0+i/BN,n=n0+i%BN;
            b_tile[i%BN][i/BN]=(k<K && n<N)?uf_mma_input<POINTWISE>(x,blockIdx.z,k,n,K,H,W,OW,KH,KW,SH,SW,PH,PW):0;
        }
        __syncthreads();
        #pragma unroll
        for(int slice=0;slice<BK;slice+=32) {
        uint32_t al[4],ah[4];
        #pragma unroll
        for(int r=0;r<4;++r)
            uf_split(&a_tile[warp*16+group+(r%2)*8][slice+quad*4+(r/2)*16],al[r],ah[r]);
        #pragma unroll
        for(int tile=0;tile<BN/8;++tile) {
            uint32_t bl[2],bh[2];
            #pragma unroll
            for(int r=0;r<2;++r) uf_split(&b_tile[tile*8+group][slice+quad*4+r*16],bl[r],bh[r]);
            uf_mma_bytes<false,false>(ll[tile],al,bl);
            uf_mma_bytes<true,false>(cross[tile],ah,bl);
            uf_mma_bytes<false,true>(cross[tile],al,bh);
            uf_mma_bytes<true,true>(hh[tile],ah,bh);
        }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int tile=0;tile<BN/8;++tile) {
        #pragma unroll
        for(int r=0;r<4;++r) {
            int m=m0+warp*16+group+(r/2)*8,n=n0+tile*8+quad*2+r%2;
            if(m<M && n<N) {
                int64_t sum=static_cast<int64_t>(ll[tile][r])+
                    static_cast<int64_t>(static_cast<int32_t>(cross[tile][r]))*256+
                    static_cast<int64_t>(static_cast<int32_t>(hh[tile][r]))*65536;
                out[(static_cast<int64_t>(blockIdx.z)*M+m)*N+n]=finish(sum,bias?bias[m]:0);
            }
        }
    }
#else
    // Correct fallback if pre-Ampere PTX is executed on any supported GPU.
    for(int i=threadIdx.x;i<BM*BN;i+=blockDim.x) {
        int m=blockIdx.y*BM+i/BN,n=blockIdx.x*BN+i%BN;
        if(m>=M || n>=N) continue;
        int64_t sum=0;
        for(int k=0;k<K;++k)
            sum+=static_cast<int64_t>(uf_mma_input<POINTWISE>(x,blockIdx.z,k,n,K,H,W,OW,KH,KW,SH,SW,PH,PW))*weight[static_cast<int64_t>(m)*K+k];
        out[(static_cast<int64_t>(blockIdx.z)*M+m)*N+n]=finish(sum,bias?bias[m]:0);
    }
#endif
}
