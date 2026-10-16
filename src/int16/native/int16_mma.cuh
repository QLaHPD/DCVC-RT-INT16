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

// Pointwise MMA consumes channel-last input directly, avoiding shared-memory
// transposes and block barriers. Inputs are packed once per convolution.
template<int BN, bool REDUCE4, int STEPS=1, int WARPS=4,
         bool OUT_LAST=false,bool RES_LAST=false,bool DUAL_OUT=false>
__global__ __launch_bounds__(WARPS*32) void uf_mma_direct(int16_t* __restrict__ out,const int16_t* __restrict__ x,
    const int16_t* __restrict__ weight,const int16_t* bias,int M,int N,int K,const int16_t* lut,
    const int16_t* residual,const int16_t* shortcut,int16_t* packed_output) {
    static_assert(!DUAL_OUT || (!REDUCE4 && !OUT_LAST),"dual output requires a plain NCHW projection");
#if __CUDA_ARCH__ >= 800
    const int warp=threadIdx.x/32,lane=threadIdx.x%32;
    const int group=lane/4,quad=lane%4,m0=blockIdx.y*(WARPS*16),n0=blockIdx.x*BN;
    uint32_t ll[BN/8][4]={},cross[BN/8][4]={},hh[BN/8][4]={};
    for(int base=0;base<K;base+=32*STEPS) {
        uint32_t al[STEPS][4]={},ah[STEPS][4]={};
        #pragma unroll
        for(int step=0;step<STEPS;++step) {
            #pragma unroll
            for(int r=0;r<4;++r) {
                int m=m0+warp*16+group+(r%2)*8;
                if(m<M && base+step*32<K)
                    uf_split(weight+static_cast<int64_t>(m)*K+base+step*32+quad*4+(r/2)*16,al[step][r],ah[step][r]);
            }
        }
        #pragma unroll
        for(int tile=0;tile<BN/8;++tile) {
            uint32_t bl[STEPS][2]={},bh[STEPS][2]={};
            int n=n0+tile*8+group;
            #pragma unroll
            for(int step=0;step<STEPS;++step) {
                #pragma unroll
                for(int r=0;r<2;++r)
                    if(n<N && base+step*32<K)
                        uf_split(x+(static_cast<int64_t>(blockIdx.z)*N+n)*K+base+step*32+quad*4+r*16,bl[step][r],bh[step][r]);
            }
            #pragma unroll
            for(int step=0;step<STEPS;++step) {
                uf_mma_bytes<false,false>(ll[tile],al[step],bl[step]);
                uf_mma_bytes<true,false>(cross[tile],ah[step],bl[step]);
                uf_mma_bytes<false,true>(cross[tile],al[step],bh[step]);
                uf_mma_bytes<true,true>(hh[tile],ah[step],bh[step]);
            }
        }
    }
    #pragma unroll
    for(int tile=0;tile<BN/8;++tile) {
        #pragma unroll
        for(int r=0;r<4;++r) {
            int m=m0+warp*16+group+(r/2)*8,n=n0+tile*8+quad*2+r%2;
            int64_t sum=static_cast<int64_t>(ll[tile][r])+
                static_cast<int64_t>(static_cast<int32_t>(cross[tile][r]))*256+
                static_cast<int64_t>(static_cast<int32_t>(hh[tile][r]))*65536;
            if constexpr(REDUCE4) {
                int value=(m<M && n<N)?lut[static_cast<int>(finish(sum,bias?bias[m]:0))+32768]:0;
                value+=__shfl_xor_sync(0xffffffff,value,4);
                value+=__shfl_xor_sync(0xffffffff,value,8);
                if(group%4==0 && m<M && n<N) {
                    int64_t index=OUT_LAST?(static_cast<int64_t>(blockIdx.z)*N+n)*(M/4)+m/4:
                        (static_cast<int64_t>(blockIdx.z)*(M/4)+m/4)*N+n;
                    out[index]=
                        value<-32768?-32768:(value>32767?32767:value);
                }
            } else if(m<M && n<N) {
                int64_t index=(static_cast<int64_t>(blockIdx.z)*M+m)*N+n;
                int64_t last=(static_cast<int64_t>(blockIdx.z)*N+n)*M+m;
                int16_t value=postprocess(finish(sum,bias?bias[m]:0),lut,residual,shortcut,
                                          index,RES_LAST?last:index);
                out[OUT_LAST?last:index]=value;
                if constexpr(DUAL_OUT) packed_output[last]=value;
            }
        }
    }
#else
    for(int i=threadIdx.x;i<(WARPS*16)*BN;i+=blockDim.x) {
        int m=blockIdx.y*(WARPS*16)+i/BN,n=blockIdx.x*BN+i%BN;
        if(m>=M || n>=N) continue;
        if constexpr(REDUCE4) { if(m%4) continue; }
        int value=0;
        for(int row=0;row<(REDUCE4?4:1);++row) {
            int64_t sum=0;
            for(int k=0;k<K;++k)
                sum+=static_cast<int64_t>(x[(static_cast<int64_t>(blockIdx.z)*N+n)*K+k])*weight[static_cast<int64_t>(m+row)*K+k];
            int16_t rounded=finish(sum,bias?bias[m+row]:0);
            if constexpr(REDUCE4) value+=lut[static_cast<int>(rounded)+32768];
            else value=rounded;
        }
        int64_t index=(static_cast<int64_t>(blockIdx.z)*(REDUCE4?M/4:M)+(REDUCE4?m/4:m))*N+n;
        int64_t last=(static_cast<int64_t>(blockIdx.z)*N+n)*(REDUCE4?M/4:M)+(REDUCE4?m/4:m);
        int16_t rounded=value<-32768?-32768:(value>32767?32767:value);
        int16_t value_out=REDUCE4?rounded:postprocess(rounded,lut,residual,shortcut,
                                                    index,RES_LAST?last:index);
        out[OUT_LAST?last:index]=value_out;
        if constexpr(DUAL_OUT) packed_output[last]=value_out;
    }
#endif
}

// FFN output channels occupy the columns of the MMA instead of its rows.
// Two adjacent channels then live in each lane, so four-channel activation
// reduction needs one warp shuffle rather than two.
template<int BN, int STEPS=1, int WARPS=4, int MIN_BLOCKS=1,bool OUT_LAST=false>
__global__ __launch_bounds__(WARPS*32,MIN_BLOCKS) void uf_mma_ffn(int16_t* __restrict__ out,
    const int16_t* __restrict__ x,const int16_t* __restrict__ weight,const int16_t* bias,
    int M,int N,int K,const int16_t* lut) {
#if __CUDA_ARCH__ >= 800
    const int warp=threadIdx.x/32,lane=threadIdx.x%32;
    const int group=lane/4,quad=lane%4,n0=blockIdx.y*(WARPS*16),m0=blockIdx.x*BN;
    uint32_t ll[BN/8][4]={},cross[BN/8][4]={},hh[BN/8][4]={};
    for(int base=0;base<K;base+=32*STEPS) {
        uint32_t al[STEPS][4]={},ah[STEPS][4]={};
        #pragma unroll
        for(int step=0;step<STEPS;++step) {
            #pragma unroll
            for(int r=0;r<4;++r) {
                int n=n0+warp*16+group+(r%2)*8;
                if(n<N && base+step*32<K)
                    uf_split(x+(static_cast<int64_t>(blockIdx.z)*N+n)*K+base+step*32+quad*4+(r/2)*16,
                             al[step][r],ah[step][r]);
            }
        }
        #pragma unroll
        for(int tile=0;tile<BN/8;++tile) {
            uint32_t bl[STEPS][2]={},bh[STEPS][2]={};
            int m=m0+tile*8+group;
            #pragma unroll
            for(int step=0;step<STEPS;++step) {
                #pragma unroll
                for(int r=0;r<2;++r)
                    if(m<M && base+step*32<K)
                        uf_split(weight+static_cast<int64_t>(m)*K+base+step*32+quad*4+r*16,
                                 bl[step][r],bh[step][r]);
            }
            #pragma unroll
            for(int step=0;step<STEPS;++step) {
                uf_mma_bytes<false,false>(ll[tile],al[step],bl[step]);
                uf_mma_bytes<true,false>(cross[tile],ah[step],bl[step]);
                uf_mma_bytes<false,true>(cross[tile],al[step],bh[step]);
                uf_mma_bytes<true,true>(hh[tile],ah[step],bh[step]);
            }
        }
    }
    #pragma unroll
    for(int tile=0;tile<BN/8;++tile) {
        #pragma unroll
        for(int row=0;row<2;++row) {
            int n=n0+warp*16+group+row*8,m=m0+tile*8+quad*2;
            int value=0;
            #pragma unroll
            for(int col=0;col<2;++col) {
                int r=row*2+col;
                int64_t sum=static_cast<int64_t>(ll[tile][r])+
                    static_cast<int64_t>(static_cast<int32_t>(cross[tile][r]))*256+
                    static_cast<int64_t>(static_cast<int32_t>(hh[tile][r]))*65536;
                if(m+col<M && n<N)
                    value+=lut[static_cast<int>(finish(sum,bias?bias[m+col]:0))+32768];
            }
            value+=__shfl_xor_sync(0xffffffff,value,1);
            if(quad%2==0 && m<M && n<N) {
                int64_t index=OUT_LAST?(static_cast<int64_t>(blockIdx.z)*N+n)*(M/4)+m/4:
                    (static_cast<int64_t>(blockIdx.z)*(M/4)+m/4)*N+n;
                out[index]=
                    value<-32768?-32768:(value>32767?32767:value);
            }
        }
    }
#else
    constexpr int BM=WARPS*16;
    for(int i=threadIdx.x;i<(BN/4)*BM;i+=blockDim.x) {
        int m=blockIdx.x*BN+(i/BM)*4,n=blockIdx.y*BM+i%BM;
        if(m>=M || n>=N) continue;
        int value=0;
        for(int row=0;row<4;++row) {
            int64_t sum=0;
            for(int k=0;k<K;++k)
                sum+=static_cast<int64_t>(x[(static_cast<int64_t>(blockIdx.z)*N+n)*K+k])*
                     weight[static_cast<int64_t>(m+row)*K+k];
            value+=lut[static_cast<int>(finish(sum,bias?bias[m+row]:0))+32768];
        }
        int64_t index=OUT_LAST?(static_cast<int64_t>(blockIdx.z)*N+n)*(M/4)+m/4:
            (static_cast<int64_t>(blockIdx.z)*(M/4)+m/4)*N+n;
        out[index]=
            value<-32768?-32768:(value>32767?32767:value);
    }
#endif
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

template<bool POINTWISE, int A_PADDING=0, int TILE_N=32, int TILE_K=32, bool REDUCE4=false,
         int B_PADDING=2>
__global__ __launch_bounds__(128) void uf_mma_conv(int16_t* out,const int16_t* x,
    const int16_t* weight,const int16_t* bias,int M,int N,int K,int H,int W,int OW,
    int KH,int KW,int SH,int SW,int PH,int PW,const int16_t* lut,
    const int16_t* residual,const int16_t* shortcut) {
    constexpr int BM=64, BN=TILE_N, BK=TILE_K, THREADS=128;
#if __CUDA_ARCH__ >= 800
    __shared__ __align__(4) int16_t a_tile[BM][BK+A_PADDING];
    __shared__ __align__(4) int16_t b_tile[BN][BK+B_PADDING];
    const int tid=threadIdx.x,warp=tid/32,lane=tid%32;
    const int group=lane/4,quad=lane%4,m0=blockIdx.y*BM,n0=blockIdx.x*BN;
    uint32_t ll[BN/8][4]={},cross[BN/8][4]={},hh[BN/8][4]={};
    for(int k0=0;k0<K;k0+=BK) {
        #pragma unroll
        for(int i=tid;i<BM*BK;i+=THREADS) {
            const int m=m0+i/BK,k=k0+i%BK;
            a_tile[i/BK][i%BK]=(m<M && k<K)?weight[static_cast<int64_t>(m)*K+k]:0;
        }
        #pragma unroll
        for(int i=tid;i<BK*BN;i+=THREADS) {
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
            int64_t sum=static_cast<int64_t>(ll[tile][r])+
                static_cast<int64_t>(static_cast<int32_t>(cross[tile][r]))*256+
                static_cast<int64_t>(static_cast<int32_t>(hh[tile][r]))*65536;
            if constexpr(REDUCE4) {
                // Round/saturate each convolution output before its LUT lookup.
                // Adjacent channel rows occupy lanes four places apart. All
                // lanes participate, including spatial and channel tail lanes.
                int value=(m<M && n<N)?lut[static_cast<int>(finish(sum,bias?bias[m]:0))+32768]:0;
                value+=__shfl_xor_sync(0xffffffff,value,4);
                value+=__shfl_xor_sync(0xffffffff,value,8);
                if(group%4==0 && m<M && n<N)
                    out[(static_cast<int64_t>(blockIdx.z)*(M/4)+m/4)*N+n]=
                        value<-32768?-32768:(value>32767?32767:value);
            } else if(m<M && n<N) {
                int64_t index=(static_cast<int64_t>(blockIdx.z)*M+m)*N+n;
                out[index]=postprocess(finish(sum,bias?bias[m]:0),lut,residual,shortcut,index);
            }
        }
    }
#else
    // Correct fallback if pre-Ampere PTX is executed on any supported GPU.
    for(int i=threadIdx.x;i<BM*BN;i+=blockDim.x) {
        int m=blockIdx.y*BM+i/BN,n=blockIdx.x*BN+i%BN;
        if(m>=M || n>=N) continue;
        if constexpr(REDUCE4) { if(m%4) continue; }
        int value=0;
        for(int row=0;row<(REDUCE4?4:1);++row) {
            int64_t sum=0;
            for(int k=0;k<K;++k)
                sum+=static_cast<int64_t>(uf_mma_input<POINTWISE>(x,blockIdx.z,k,n,K,H,W,OW,KH,KW,SH,SW,PH,PW))*weight[static_cast<int64_t>(m+row)*K+k];
            int16_t rounded=finish(sum,bias?bias[m+row]:0);
            if constexpr(REDUCE4) value+=lut[static_cast<int>(rounded)+32768];
            else value=rounded;
        }
        int64_t index=(static_cast<int64_t>(blockIdx.z)*(REDUCE4?M/4:M)+(REDUCE4?m/4:m))*N+n;
        int16_t rounded=value<-32768?-32768:(value>32767?32767:value);
        out[index]=REDUCE4?rounded:postprocess(rounded,lut,residual,shortcut,index);
    }
#endif
}
