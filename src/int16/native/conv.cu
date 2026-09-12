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

template<int A_PADDING=0, int TILE_N=32, int TILE_K=32>
static torch::Tensor uf_conv_impl(const torch::Tensor& input, const torch::Tensor& weights,
    const c10::optional<torch::Tensor>& bias, int sh,int sw,int ph,int pw,int groups,bool use_mma) {
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
    auto out=torch::empty({x.size(0),co,oh,ow},x.options());
    torch::Tensor bias_contiguous;
    const int16_t* ptr=nullptr;
    if(bias.has_value()) {
        TORCH_CHECK(bias->device()==x.device() && bias->scalar_type()==at::kShort && bias->numel()==co && bias->dim()==1,"invalid bias");
        bias_contiguous=bias->contiguous(); ptr=bias_contiguous.data_ptr<int16_t>();
    }
    auto stream=c10::cuda::getCurrentCUDAStream();
    bool point=kh==1 && kw==1 && sh==1 && sw==1 && ph==0 && pw==0;
    int64_t reduction=static_cast<int64_t>(ci)*kh*kw;
    if(use_mma && groups==1 && reduction<=32768) {
        dim3 grid((oh*ow+TILE_N-1)/TILE_N,(co+63)/64,x.size(0));
        if(point)
            uf_mma_conv<true,A_PADDING,TILE_N,TILE_K><<<grid,128,0,stream>>>(out.data_ptr<int16_t>(),x.data_ptr<int16_t>(),
                weight.data_ptr<int16_t>(),ptr,co,oh*ow,reduction,h,w,ow,kh,kw,sh,sw,ph,pw);
        else
            uf_mma_conv<false,A_PADDING,TILE_N,TILE_K><<<grid,128,0,stream>>>(out.data_ptr<int16_t>(),x.data_ptr<int16_t>(),
                weight.data_ptr<int16_t>(),ptr,co,oh*ow,reduction,h,w,ow,kh,kw,sh,sw,ph,pw);
    } else if(point && groups==1)
        pointwise<<<dim3((h*w+15)/16,(co+15)/16,x.size(0)),dim3(16,16),0,stream>>>(
            out.data_ptr<int16_t>(),x.data_ptr<int16_t>(),weight.data_ptr<int16_t>(),ptr,h*w,ci,co);
    else
        general<<<(out.numel()+255)/256,256,0,stream>>>(out.data_ptr<int16_t>(),x.data_ptr<int16_t>(),
            weight.data_ptr<int16_t>(),ptr,out.numel(),ci,co,h,w,oh,ow,kh,kw,sh,sw,ph,pw,groups);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor uf_conv(const torch::Tensor& x,const torch::Tensor& w,
    const c10::optional<torch::Tensor>& b,int sh,int sw,int ph,int pw,int g) {
    if(x.dim()==4 && w.dim()==4 && w.size(2)==1 && w.size(3)==1 &&
       sh==1 && sw==1 && ph==0 && pw==0 && g==1) {
        // Wider spatial tiles reuse weights on large, exactly tiled planes.
        // Otherwise reduce barrier frequency with 64-element K tiles.
        const int64_t spatial=x.size(2)*x.size(3);
        if(spatial>=256 && spatial%64==0 && x.size(1)<=1024 && w.size(0)<=x.size(1))
            return uf_conv_impl<16,64,32>(x,w,b,sh,sw,ph,pw,g,true);
        return uf_conv_impl<16,32,64>(x,w,b,sh,sw,ph,pw,g,true);
    }
    return uf_conv_impl(x,w,b,sh,sw,ph,pw,g,true);
}
torch::Tensor uf_conv_generic(const torch::Tensor& x,const torch::Tensor& w,
    const c10::optional<torch::Tensor>& b,int sh,int sw,int ph,int pw,int g) {
    return uf_conv_impl(x,w,b,sh,sw,ph,pw,g,false);
}
torch::Tensor uf_conv_baseline(const torch::Tensor& x,const torch::Tensor& w,
    const c10::optional<torch::Tensor>& b,int sh,int sw,int ph,int pw,int g) {
    return uf_conv_impl(x,w,b,sh,sw,ph,pw,g,true);
}
