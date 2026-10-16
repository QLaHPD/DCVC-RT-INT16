// Fused integer operations avoid full-sized int64 temporary feature maps.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {
__device__ int16_t sat(int64_t x) { return x < -32768 ? -32768 : (x > 32767 ? 32767 : x); }
__device__ int64_t rna(int64_t x, int d) { return x < 0 ? -((-x+d/2)/d) : (x+d/2)/d; }
void check(const torch::Tensor& x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type()==at::kShort && x.is_contiguous(), "contiguous CUDA INT16 required");
    TORCH_CHECK(x.numel()>0 && x.numel()<=INT32_MAX, "unsupported tensor size");
}
__global__ void add_kernel(int16_t* out,const int16_t* a,const int16_t* b,
    const int16_t* c,const int16_t* d,int64_t n) {
    int64_t i=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n) out[i]=sat(static_cast<int32_t>(a[i])+b[i]+(c?c[i]:0)+(d?d[i]:0));
}
template<int MODE>
__global__ void mul_kernel(int16_t* out,const int16_t* x,const int16_t* scale,int n,
    int c,int h,int w,int sb,int sc,int sh,int sw) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    int offset;
    if constexpr(MODE==0) offset=0;
    else if constexpr(MODE==1) offset=i;
    else if constexpr(MODE==2) offset=sb==1?(i/(w*h))%c:i/(w*h);
    else {
        int iw=i%w, ih=(i/w)%h, ic=(i/(w*h))%c, ib=i/(w*h*c);
        offset=(((sb==1?0:ib)*sc+(sc==1?0:ic))*sh+(sh==1?0:ih))*sw+(sw==1?0:iw);
    }
    // Every INT16 product and its rounding offset fit a signed INT32.
    int product=static_cast<int>(x[i])*scale[offset];
    int rounded=product<0?-((-product+256)/512):(product+256)/512;
    out[i]=sat(rounded);
}
template<class T> __global__ void lut_kernel(T* out,const int16_t* x,const T* lut,int64_t n) {
    int64_t i=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n) out[i]=lut[static_cast<int32_t>(x[i])+32768];
}
__global__ void bytes_kernel(int16_t* out,const uint8_t* x,int64_t n) {
    int64_t i=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n) out[i]=rna((2*static_cast<int32_t>(x[i])-255)*512,510);
}
__global__ void symbols_kernel(int8_t* out,const int16_t* x,int64_t n) {
    int64_t i=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n) out[i]=rna(x[i],512);
}
__global__ void from_symbols_kernel(int16_t* out,const int8_t* x,int64_t n) {
    int64_t i=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n) out[i]=sat(static_cast<int>(x[i])*512);
}
__global__ void reciprocal_kernel(int16_t* out,const int16_t* x,int64_t n) {
    int64_t i=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n) {
        int denominator=x[i]<256?256:x[i];
        out[i]=(262144+denominator/2)/denominator;
    }
}
__global__ void wsilu4_kernel(int16_t* out,const int16_t* x,const int16_t* lut,int64_t n,int channels,int hw) {
    int64_t i=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=n) return;
    int pos=i%hw, ch=(i/hw)%channels, batch=i/(static_cast<int64_t>(hw)*channels);
    int64_t first=(static_cast<int64_t>(batch)*channels*4+ch*4)*hw+pos;
    int32_t sum=0;
    #pragma unroll
    for(int k=0;k<4;++k) sum+=lut[static_cast<int32_t>(x[first+k*hw])+32768];
    out[i]=sat(sum);
}
__global__ void raw_frame_kernel(uint8_t* out,const int16_t* x,int count,int width,int height,
    int input_width,int input_height) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count) return;
    int pixels=width*height,value;
    if(i<pixels) {
        int offset=(i/width)*input_width+i%width;
        value=rna((static_cast<int>(x[offset])+256)*255,512);
    } else {
        int uv=i-pixels,plane=uv/(pixels/4),position=uv%(pixels/4);
        int row=(position/(width/2))*2,col=(position%(width/2))*2;
        int offset=(plane+1)*input_width*input_height+row*input_width+col;
        int sum=static_cast<int>(x[offset])+x[offset+1]+x[offset+input_width]+x[offset+input_width+1];
        int numerator=(sum+1024)*255;
        value=numerator<0?-((-numerator+2047)/2048):numerator/2048;
    }
    out[i]=value<0?0:(value>255?255:value);
}
}

torch::Tensor uf_add(const std::vector<torch::Tensor>& values) {
    TORCH_CHECK(values.size()>=2 && values.size()<=4,"add supports 2 to 4 tensors");
    auto x=values[0].contiguous(); check(x); c10::cuda::CUDAGuard guard(x.device());
    std::vector<torch::Tensor> operands;
    for(const auto& value:values) {
        TORCH_CHECK(value.device()==x.device() && value.sizes()==x.sizes(),"add shape/device mismatch");
        operands.push_back(value.contiguous()); check(operands.back());
    }
    auto out=torch::empty_like(x);
    add_kernel<<<(x.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(out.data_ptr<int16_t>(),
        operands[0].data_ptr<int16_t>(),operands[1].data_ptr<int16_t>(),
        values.size()>2?operands[2].data_ptr<int16_t>():nullptr,
        values.size()>3?operands[3].data_ptr<int16_t>():nullptr,x.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return out;
}
torch::Tensor uf_mul(const torch::Tensor& input,const torch::Tensor& factor) {
    auto x=input.contiguous(), scale=factor.contiguous(); check(x); check(scale);
    TORCH_CHECK(x.device()==scale.device() && x.dim()==4 && scale.dim()==4,"multiply requires matching CUDA 4D tensors");
    for(int d=0;d<4;++d) TORCH_CHECK(scale.size(d)==1 || scale.size(d)==x.size(d),"invalid broadcast");
    c10::cuda::CUDAGuard guard(x.device()); auto out=torch::empty_like(x);
    auto stream=c10::cuda::getCurrentCUDAStream();
#define UF_MULTIPLY(MODE) \
    mul_kernel<MODE><<<(x.numel()+255)/256,256,0,stream>>>(out.data_ptr<int16_t>(), \
        x.data_ptr<int16_t>(),scale.data_ptr<int16_t>(),x.numel(),x.size(1),x.size(2),x.size(3), \
        scale.size(0),scale.size(1),scale.size(2),scale.size(3))
    if(scale.numel()==1) { UF_MULTIPLY(0); }
    else if(scale.numel()==x.numel()) { UF_MULTIPLY(1); }
    else if(scale.size(1)==x.size(1) && scale.size(2)==1 && scale.size(3)==1) { UF_MULTIPLY(2); }
    else { UF_MULTIPLY(3); }
#undef UF_MULTIPLY
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return out;
}
torch::Tensor uf_lut(const torch::Tensor& input,const torch::Tensor& table) {
    auto x=input.contiguous(), lut=table.contiguous(); check(x);
    TORCH_CHECK(lut.device()==x.device() && lut.dim()==1 && lut.numel()==65536,"invalid lookup table");
    TORCH_CHECK(lut.scalar_type()==at::kShort || lut.scalar_type()==at::kByte,"invalid table dtype");
    c10::cuda::CUDAGuard guard(x.device()); auto out=torch::empty(x.sizes(),x.options().dtype(lut.scalar_type()));
    if(lut.scalar_type()==at::kShort)
        lut_kernel<<<(x.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(out.data_ptr<int16_t>(),x.data_ptr<int16_t>(),lut.data_ptr<int16_t>(),x.numel());
    else
        lut_kernel<<<(x.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(out.data_ptr<uint8_t>(),x.data_ptr<int16_t>(),lut.data_ptr<uint8_t>(),x.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return out;
}
torch::Tensor uf_bytes(const torch::Tensor& input) {
    TORCH_CHECK(input.is_cuda() && input.scalar_type()==at::kByte && input.numel()>0 && input.numel()<=INT32_MAX,"CUDA bytes required");
    c10::cuda::CUDAGuard guard(input.device()); auto x=input.contiguous();
    auto out=torch::empty(x.sizes(),x.options().dtype(at::kShort));
    bytes_kernel<<<(x.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(out.data_ptr<int16_t>(),x.data_ptr<uint8_t>(),x.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return out;
}
torch::Tensor uf_wsilu4(const torch::Tensor& input,const torch::Tensor& table) {
    auto x=input.contiguous(),lut=table.contiguous(); check(x); check(lut);
    TORCH_CHECK(x.dim()==4 && x.size(1)%4==0 && lut.dim()==1 && lut.numel()==65536 && lut.device()==x.device(),"invalid WSiLU4 inputs");
    c10::cuda::CUDAGuard guard(x.device()); auto out=torch::empty({x.size(0),x.size(1)/4,x.size(2),x.size(3)},x.options());
    wsilu4_kernel<<<(out.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(out.data_ptr<int16_t>(),x.data_ptr<int16_t>(),lut.data_ptr<int16_t>(),out.numel(),out.size(1),x.size(2)*x.size(3));
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return out;
}

torch::Tensor uf_raw_frame(const torch::Tensor& input,int width,int height) {
    auto x=input.contiguous(); check(x);
    TORCH_CHECK(x.dim()==4 && x.size(0)==1 && x.size(1)==3 && width>0 && height>0 &&
                width%2==0 && height%2==0 && width<=x.size(3) && height<=x.size(2),"invalid YUV420 dimensions");
    c10::cuda::CUDAGuard guard(x.device());
    int64_t count=static_cast<int64_t>(width)*height*3/2;
    auto out=torch::empty({count},x.options().dtype(at::kByte));
    raw_frame_kernel<<<(count+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(out.data_ptr<uint8_t>(),
        x.data_ptr<int16_t>(),count,width,height,x.size(3),x.size(2));
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return out;
}

torch::Tensor uf_symbols(const torch::Tensor& input) {
    auto x=input.contiguous(); check(x);
    c10::cuda::CUDAGuard guard(x.device());
    auto out=torch::empty(x.sizes(),x.options().dtype(at::kChar));
    symbols_kernel<<<(x.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
        out.data_ptr<int8_t>(),x.data_ptr<int16_t>(),x.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return out;
}

torch::Tensor uf_from_symbols(const torch::Tensor& input) {
    TORCH_CHECK(input.is_cuda() && input.scalar_type()==at::kChar && input.numel()>0 &&
                input.numel()<=INT32_MAX,"CUDA int8 symbols required");
    c10::cuda::CUDAGuard guard(input.device()); auto x=input.contiguous();
    auto out=torch::empty(x.sizes(),x.options().dtype(at::kShort));
    from_symbols_kernel<<<(x.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
        out.data_ptr<int16_t>(),x.data_ptr<int8_t>(),x.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return out;
}

torch::Tensor uf_reciprocal(const torch::Tensor& input) {
    auto x=input.contiguous(); check(x);
    c10::cuda::CUDAGuard guard(x.device()); auto out=torch::empty_like(x);
    reciprocal_kernel<<<(x.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
        out.data_ptr<int16_t>(),x.data_ptr<int16_t>(),x.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return out;
}
