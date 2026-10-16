// UF integer arithmetic v1: int16 storage, exact int64 dot products.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

__device__ int16_t finish(int64_t sum, int16_t bias) {
    int64_t rounded = sum < 0 ? -((-sum + 4096) / 8192) : (sum + 4096) / 8192;
    rounded += bias;
    return static_cast<int16_t>(rounded < -32768 ? -32768 : (rounded > 32767 ? 32767 : rounded));
}

__device__ int16_t postprocess(int16_t value,const int16_t* lut,
    const int16_t* residual,const int16_t* shortcut,int64_t index) {
    if(lut) value=lut[static_cast<int>(value)+32768];
    if(residual) {
        int sum=static_cast<int>(value)+residual[index];
        value=sum<-32768?-32768:(sum>32767?32767:sum);
    }
    if(shortcut) {
        int sum=static_cast<int>(value)+shortcut[index];
        value=sum<-32768?-32768:(sum>32767?32767:sum);
    }
    return value;
}

#include "int16_mma.cuh"

__global__ void pointwise(int16_t* out, const int16_t* x, const int16_t* weight,
                          const int16_t* bias, int n, int ci, int co) {
    __shared__ int16_t a[16][32];
    __shared__ int16_t b[32][16];
    const int col = blockIdx.x * 16 + threadIdx.x;
    const int row = blockIdx.y * 16 + threadIdx.y;
    const int tid = threadIdx.y * 16 + threadIdx.x;
    int64_t sum = 0;
    for (int base = 0; base < ci; base += 32) {
        for (int j = tid; j < 512; j += 256) {
            const int m = j / 32, k = j % 32;
            const int channel = blockIdx.y * 16 + m;
            a[m][k] = channel < co && base+k < ci ? weight[channel*ci+base+k] : 0;
            const int kb = j / 16, nb = blockIdx.x * 16 + j % 16;
            b[kb][j % 16] = base+kb < ci && nb < n ? x[(blockIdx.z*ci+base+kb)*n+nb] : 0;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < 32; ++k)
            sum += static_cast<int64_t>(a[threadIdx.y][k]) * b[k][threadIdx.x];
        __syncthreads();
    }
    if (row < co && col < n)
        out[(blockIdx.z*co+row)*n+col] = finish(sum, bias ? bias[row] : 0);
}

__global__ void general(int16_t* out, const int16_t* x, const int16_t* weight,
    const int16_t* bias, int64_t count, int ci, int co, int h, int w,
    int oh, int ow, int kh, int kw, int sh, int sw, int ph, int pw, int groups) {
    int64_t index = static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (index >= count) return;
    int ox=index%ow, oy=(index/ow)%oh, oc=(index/(ow*oh))%co;
    int batch=index/(static_cast<int64_t>(ow)*oh*co), cg=ci/groups;
    int first=(oc/(co/groups))*cg;
    int64_t sum=0;
    for (int c=0;c<cg;++c) for(int ky=0;ky<kh;++ky) for(int kx=0;kx<kw;++kx) {
        int iy=oy*sh+ky-ph, ix=ox*sw+kx-pw;
        if(iy>=0 && iy<h && ix>=0 && ix<w)
            sum += static_cast<int64_t>(x[((batch*ci+first+c)*h+iy)*w+ix]) *
                   weight[((oc*cg+c)*kh+ky)*kw+kx];
    }
    out[index]=finish(sum,bias?bias[oc]:0);
}

__global__ void unfold_int16(int16_t* out,const int16_t* x,int count,int ci,int h,int w,
    int oh,int ow,int kh,int kw,int sh,int sw,int ph,int pw) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count) return;
    int n=i%(oh*ow),k=(i/(oh*ow))%(ci*kh*kw),batch=i/(ci*kh*kw*oh*ow);
    int iy=(n/ow)*sh+(k%(kh*kw))/kw-ph,ix=(n%ow)*sw+k%kw-pw;
    out[i]=(iy>=0 && iy<h && ix>=0 && ix<w)?x[((batch*ci+k/(kh*kw))*h+iy)*w+ix]:0;
}

__global__ void depthwise3x3(int16_t* out,const int16_t* x,const int16_t* weight,
    const int16_t* bias,int count,int channels,int h,int w) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=count) return;
    int col=index%w,row=(index/w)%h,channel=(index/(h*w))%channels;
    int64_t sum=0;
    #pragma unroll
    for(int ky=0;ky<3;++ky) {
        int iy=row+ky-1;
        #pragma unroll
        for(int kx=0;kx<3;++kx) {
            int ix=col+kx-1;
            if(iy>=0 && iy<h && ix>=0 && ix<w) {
                int product=static_cast<int>(x[index+(ky-1)*w+kx-1])*weight[channel*9+ky*3+kx];
                sum+=product;
            }
        }
    }
    out[index]=finish(sum,bias?bias[channel]:0);
}

template<int A_PADDING=0, int TILE_N=32, int TILE_K=32, bool REDUCE4=false, bool PACK=false,
         int B_PADDING=2, bool DIRECT=false, int DIRECT_STEPS=1, int DIRECT_WARPS=4>
static torch::Tensor uf_conv_impl(const torch::Tensor& input, const torch::Tensor& weights,
    const c10::optional<torch::Tensor>& bias, int sh,int sw,int ph,int pw,int groups,bool use_mma,
    const c10::optional<torch::Tensor>& table=c10::nullopt,
    const c10::optional<torch::Tensor>& residual=c10::nullopt,
    const c10::optional<torch::Tensor>& shortcut=c10::nullopt) {
    TORCH_CHECK(input.is_cuda() && weights.device()==input.device(), "CUDA device mismatch");
    TORCH_CHECK(input.scalar_type()==at::kShort && weights.scalar_type()==at::kShort, "int16 required");
    TORCH_CHECK(input.dim()==4 && weights.dim()==4, "4D tensors required");
    TORCH_CHECK(input.numel()<=INT32_MAX && weights.numel()<=INT32_MAX,"convolution tensors exceed indexing bounds");
    TORCH_CHECK(sh>0 && sw>0 && ph>=0 && pw>=0 && groups>0, "invalid convolution geometry");
    TORCH_CHECK(weights.size(1)*groups==input.size(1) && weights.size(0)%groups==0, "invalid groups");
    c10::cuda::CUDAGuard guard(input.device());
    auto x=input.contiguous(), weight=weights.contiguous();
    int ci=x.size(1),co=weight.size(0),h=x.size(2),w=x.size(3),kh=weight.size(2),kw=weight.size(3);
    TORCH_CHECK(ci>0 && co>0 && kh>0 && kw>0 && input.size(0)>0, "empty convolution");
    TORCH_CHECK(h+2*ph>=kh && w+2*pw>=kw, "kernel exceeds input");
    // Bound the maximal dot product before int64 accumulation.
    TORCH_CHECK(weight.size(1)*static_cast<int64_t>(kh)*kw < (INT64_MAX/1073741824LL)-1, "dot product overflow");
    int oh=(h+2*ph-kh)/sh+1,ow=(w+2*pw-kw)/sw+1;
    TORCH_CHECK(x.size(0)*static_cast<int64_t>(co)*oh*ow<=INT32_MAX,"convolution output exceeds indexing bounds");
    const int16_t* lut=nullptr;
    if constexpr(REDUCE4) {
        TORCH_CHECK(co%4==0 && groups==1 && kh==1 && kw==1 && sh==1 && sw==1 && ph==0 && pw==0 && ci<=32768,
                    "fused WSiLU4 requires a pointwise convolution with groups of four output channels");
        TORCH_CHECK(table.has_value() && !residual.has_value() && !shortcut.has_value(),"invalid fused WSiLU4 epilogue");
    }
    if(table.has_value()) {
        TORCH_CHECK(table->device()==x.device() && table->scalar_type()==at::kShort &&
                   table->dim()==1 && table->numel()==65536 && table->is_contiguous(), "invalid activation table");
        lut=table->data_ptr<int16_t>();
    }
    torch::Tensor bias_contiguous;
    const int16_t* ptr=nullptr;
    if(bias.has_value()) {
        TORCH_CHECK(bias->device()==x.device() && bias->scalar_type()==at::kShort && bias->numel()==co && bias->dim()==1,"invalid bias");
        bias_contiguous=bias->contiguous(); ptr=bias_contiguous.data_ptr<int16_t>();
    }
    auto stream=c10::cuda::getCurrentCUDAStream();
    bool point=kh==1 && kw==1 && sh==1 && sw==1 && ph==0 && pw==0;
    int64_t reduction=static_cast<int64_t>(ci)*kh*kw;
    if constexpr(PACK) {
        // Materialize small receptive fields once instead of redoing spatial
        // division/padding for every output-channel tile. Bound scratch to 8 MiB.
        int64_t elements=x.size(0)*reduction*oh*ow;
        if(use_mma && !point && groups==1 && co>=128 && reduction>=256 &&
           reduction<=32768 && elements<=4*1024*1024) {
            auto packed=torch::empty({x.size(0),reduction,oh,ow},x.options());
            unfold_int16<<<(elements+255)/256,256,0,stream>>>(packed.data_ptr<int16_t>(),
                x.data_ptr<int16_t>(),elements,ci,h,w,oh,ow,kh,kw,sh,sw,ph,pw);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return uf_conv_impl<16,32,64,false,false,8>(packed,weight.view({co,reduction,1,1}),bias,
                                          1,1,0,0,1,true);
        }
    }
    if constexpr(DIRECT) {
        TORCH_CHECK(point && groups==1 && ci%32==0 && ci<=32768 && x.numel()<=8*1024*1024,"direct MMA geometry");
        // A contiguous INT16 view may still start at a two-byte storage offset.
        // Direct fragment loads read uint32 pairs; shared staging handles such
        // views without imposing a stronger alignment contract on the caller.
        if((reinterpret_cast<uintptr_t>(x.data_ptr<int16_t>()) |
            reinterpret_cast<uintptr_t>(weight.data_ptr<int16_t>())) & 3)
            return uf_conv_impl<16,32,64,REDUCE4,false,8>(x,weight,bias,sh,sw,ph,pw,groups,
                                                         use_mma,table,residual,shortcut);
    }
    auto out=torch::empty({x.size(0),REDUCE4?co/4:co,oh,ow},x.options());
    torch::Tensor residual_contiguous,shortcut_contiguous;
    const int16_t* residual_ptr=nullptr;
    const int16_t* shortcut_ptr=nullptr;
    for(const auto* tensor:{&residual,&shortcut}) {
        if(tensor->has_value())
            TORCH_CHECK((*tensor)->device()==x.device() && (*tensor)->scalar_type()==at::kShort &&
                        (*tensor)->sizes()==out.sizes(),"invalid residual");
    }
    if(residual.has_value()) {
        residual_contiguous=residual->contiguous();
        residual_ptr=residual_contiguous.data_ptr<int16_t>();
    }
    if(shortcut.has_value()) {
        shortcut_contiguous=shortcut->contiguous();
        shortcut_ptr=shortcut_contiguous.data_ptr<int16_t>();
    }
    if constexpr(DIRECT) {
        auto channels_last=x.permute({0,2,3,1}).contiguous();
        constexpr int bm=DIRECT_WARPS*16;
        dim3 grid((oh*ow+TILE_N-1)/TILE_N,(co+bm-1)/bm,x.size(0));
        uf_mma_direct<TILE_N,REDUCE4,DIRECT_STEPS,DIRECT_WARPS><<<grid,DIRECT_WARPS*32,0,stream>>>(out.data_ptr<int16_t>(),channels_last.data_ptr<int16_t>(),
            weight.data_ptr<int16_t>(),ptr,co,oh*ow,ci,lut,residual_ptr,shortcut_ptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }
    if(use_mma && groups==1 && reduction<=32768) {
        dim3 grid((oh*ow+TILE_N-1)/TILE_N,(co+63)/64,x.size(0));
        if(point)
            uf_mma_conv<true,A_PADDING,TILE_N,TILE_K,REDUCE4,B_PADDING><<<grid,128,0,stream>>>(out.data_ptr<int16_t>(),x.data_ptr<int16_t>(),
                weight.data_ptr<int16_t>(),ptr,co,oh*ow,reduction,h,w,ow,kh,kw,sh,sw,ph,pw,lut,residual_ptr,shortcut_ptr);
        else
            uf_mma_conv<false,A_PADDING,TILE_N,TILE_K,REDUCE4,B_PADDING><<<grid,128,0,stream>>>(out.data_ptr<int16_t>(),x.data_ptr<int16_t>(),
                weight.data_ptr<int16_t>(),ptr,co,oh*ow,reduction,h,w,ow,kh,kw,sh,sw,ph,pw,lut,residual_ptr,shortcut_ptr);
    } else if(point && groups==1)
        pointwise<<<dim3((h*w+15)/16,(co+15)/16,x.size(0)),dim3(16,16),0,stream>>>(
            out.data_ptr<int16_t>(),x.data_ptr<int16_t>(),weight.data_ptr<int16_t>(),ptr,h*w,ci,co);
    else if(use_mma && groups==ci && co==ci && kh==3 && kw==3 && sh==1 && sw==1 && ph==1 && pw==1)
        depthwise3x3<<<(out.numel()+255)/256,256,0,stream>>>(out.data_ptr<int16_t>(),x.data_ptr<int16_t>(),
            weight.data_ptr<int16_t>(),ptr,out.numel(),ci,h,w);
    else
        general<<<(out.numel()+255)/256,256,0,stream>>>(out.data_ptr<int16_t>(),x.data_ptr<int16_t>(),
            weight.data_ptr<int16_t>(),ptr,out.numel(),ci,co,h,w,oh,ow,kh,kw,sh,sw,ph,pw,groups);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

template<bool REDUCE4>
static torch::Tensor pointwise_tiled(const torch::Tensor&,const torch::Tensor&,
    const c10::optional<torch::Tensor>&,const c10::optional<torch::Tensor>&,int,
    const c10::optional<torch::Tensor>&,const c10::optional<torch::Tensor>&);

static int pointwise_tile(const torch::Tensor& x,const torch::Tensor& w,bool reduce4=false) {
    TORCH_CHECK(x.dim()==4 && w.dim()==4,"pointwise tensors required");
    const int64_t ci=x.size(1),co=w.size(0),spatial=x.size(2)*x.size(3);
    // Packing is amortized by larger matrix products. Keep scratch <= 16 MiB.
    if(ci>=256 && ci%32==0 && ci<=32768 && x.numel()<=8*1024*1024 && spatial>=128) {
        // Two reduction fragments are prefetched together. Keep the smaller
        // live register set for the four-channel activation reduction.
        if(!reduce4) {
            if(spatial>=256 && spatial<=8192 && (ci>=384 || co>=ci*2)) return 10;
            if(spatial<256 && ci>=640) return 9;
            if(spatial<256 && ci>=384 && co<=ci) return 11;
        } else if(ci==768 && spatial<256 && co>=ci*2) return 8;
        if(co>=ci*2) return spatial>=256 && spatial<=1024?8:7;
        if(ci>=384 && spatial>1024 && spatial<=8192) return 7;
    }
    if(spatial>=256 && spatial<=1024 && spatial%64==0 && ci>=384 && ci<=1024 && co<=ci)
        return 4;
    return 6;
}

torch::Tensor uf_conv(const torch::Tensor& x,const torch::Tensor& w,
    const c10::optional<torch::Tensor>& b,int sh,int sw,int ph,int pw,int g) {
    if(x.dim()==4 && w.dim()==4 && w.size(2)==1 && w.size(3)==1 &&
       sh==1 && sw==1 && ph==0 && pw==0 && g==1) {
        return pointwise_tiled<false>(x,w,b,c10::nullopt,pointwise_tile(x,w),c10::nullopt,c10::nullopt);
    }
    return uf_conv_impl<0,32,32,false,true>(x,w,b,sh,sw,ph,pw,g,true);
}
torch::Tensor uf_conv_generic(const torch::Tensor& x,const torch::Tensor& w,
    const c10::optional<torch::Tensor>& b,int sh,int sw,int ph,int pw,int g) {
    return uf_conv_impl(x,w,b,sh,sw,ph,pw,g,false);
}
torch::Tensor uf_conv_baseline(const torch::Tensor& x,const torch::Tensor& w,
    const c10::optional<torch::Tensor>& b,int sh,int sw,int ph,int pw,int g) {
    return uf_conv_impl(x,w,b,sh,sw,ph,pw,g,true);
}

template<bool REDUCE4>
static torch::Tensor pointwise_tiled(const torch::Tensor& x,const torch::Tensor& w,
    const c10::optional<torch::Tensor>& b,const c10::optional<torch::Tensor>& lut,int tile,
    const c10::optional<torch::Tensor>& residual,const c10::optional<torch::Tensor>& shortcut) {
    TORCH_CHECK(x.dim()==4 && w.dim()==4 && w.size(2)==1 && w.size(3)==1,"pointwise tensors required");
    switch(tile) {
    case 1: return uf_conv_impl<16,16,64,REDUCE4>(x,w,b,1,1,0,0,1,true,lut,residual,shortcut);
    case 2: return uf_conv_impl<16,32,64,REDUCE4>(x,w,b,1,1,0,0,1,true,lut,residual,shortcut);
    case 3: return uf_conv_impl<16,64,32,REDUCE4>(x,w,b,1,1,0,0,1,true,lut,residual,shortcut);
    case 4: return uf_conv_impl<16,64,64,REDUCE4>(x,w,b,1,1,0,0,1,true,lut,residual,shortcut);
    case 5: return uf_conv_impl<16,32,128,REDUCE4>(x,w,b,1,1,0,0,1,true,lut,residual,shortcut);
    case 6: return uf_conv_impl<16,32,64,REDUCE4,false,8>(x,w,b,1,1,0,0,1,true,lut,residual,shortcut);
    case 7: return uf_conv_impl<16,32,64,REDUCE4,false,2,true>(x,w,b,1,1,0,0,1,true,lut,residual,shortcut);
    case 8: return uf_conv_impl<16,64,64,REDUCE4,false,2,true>(x,w,b,1,1,0,0,1,true,lut,residual,shortcut);
    case 9: return uf_conv_impl<16,32,64,REDUCE4,false,2,true,2>(x,w,b,1,1,0,0,1,true,lut,residual,shortcut);
    case 10: return uf_conv_impl<16,64,64,REDUCE4,false,2,true,2>(x,w,b,1,1,0,0,1,true,lut,residual,shortcut);
    case 11: return uf_conv_impl<16,32,64,REDUCE4,false,2,true,1,2>(x,w,b,1,1,0,0,1,true,lut,residual,shortcut);
    default: TORCH_CHECK(false,"unsupported pointwise tile");
    }
}

torch::Tensor uf_conv_tiled(const torch::Tensor& x,const torch::Tensor& w,
    const c10::optional<torch::Tensor>& b,int tile) {
    return pointwise_tiled<false>(x,w,b,c10::nullopt,tile,c10::nullopt,c10::nullopt);
}

torch::Tensor uf_conv_wsilu4(const torch::Tensor& x,const torch::Tensor& w,
    const c10::optional<torch::Tensor>& b,const torch::Tensor& lut,int tile) {
    return pointwise_tiled<true>(x,w,b,lut,tile?tile:pointwise_tile(x,w,true),c10::nullopt,c10::nullopt);
}

torch::Tensor uf_conv_post(const torch::Tensor& x,const torch::Tensor& w,
    const c10::optional<torch::Tensor>& b,const c10::optional<torch::Tensor>& lut,
    const c10::optional<torch::Tensor>& residual,const c10::optional<torch::Tensor>& shortcut,int tile) {
    TORCH_CHECK(x.dim()==4 && x.size(1)<=32768,"fused pointwise reduction exceeds MMA bounds");
    return pointwise_tiled<false>(x,w,b,lut,tile?tile:pointwise_tile(x,w),residual,shortcut);
}

torch::Tensor uf_depth_block(const torch::Tensor& x,const std::vector<torch::Tensor>& weights,
    const std::vector<c10::optional<torch::Tensor>>& biases,const torch::Tensor& lut,bool shortcut) {
    TORCH_CHECK(weights.size()==5 && biases.size()==5,"five block convolutions required");
    const auto& depth=weights[1];
    TORCH_CHECK(depth.dim()==4 && depth.size(1)==1 && depth.size(2)==3 && depth.size(3)==3,
                "3x3 depthwise block required");
    auto first=uf_conv_post(x,weights[0],biases[0],lut,c10::nullopt,c10::nullopt,0);
    auto spatial=uf_conv(first,depth,biases[1],1,1,1,1,depth.size(0));
    auto residual=uf_conv_post(spatial,weights[2],biases[2],c10::nullopt,x,c10::nullopt,0);
    auto expanded=uf_conv_wsilu4(residual,weights[3],biases[3],lut,0);
    return uf_conv_post(expanded,weights[4],biases[4],c10::nullopt,residual,
                        shortcut?c10::optional<torch::Tensor>(x):c10::nullopt,0);
}
