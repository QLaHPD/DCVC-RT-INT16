// Fused integer operations avoid full-sized int64 temporary feature maps.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {
__device__ int16_t sat(int64_t x) { return x < -32768 ? -32768 : (x > 32767 ? 32767 : x); }
// Byte conversion, INT16 symbol quantization and YUV conversion all fit INT32,
// including their rounding offsets.
__device__ int32_t rna(int32_t x, int d) { return x < 0 ? -((-x+d/2)/d) : (x+d/2)/d; }
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
__global__ void bytes_unshuffle_kernel(int16_t* out,const uint8_t* x,
    int h,int w,int oh,int ow,int tiles_w) {
    // Coalesce the byte reads and the unshuffled INT16 writes. Four padding
    // words make both sides of the 8-by-32 transpose free of bank conflicts.
    __shared__ int32_t tile[64][36];
    int ox=(blockIdx.x%tiles_w)*32,group=blockIdx.x/tiles_w;
    int oy=group%oh,plane=group/oh;
    int position=threadIdx.x/8,sub_x=threadIdx.x%8;
    if(ox+position<ow) {
        int ix=ox*8+threadIdx.x;
        ix=ix<w?ix:w-1;
        #pragma unroll
        for(int dy=0;dy<8;++dy) {
            int iy=oy*8+dy;
            iy=iy<h?iy:h-1;
            int64_t offset=(static_cast<int64_t>(plane)*h+iy)*w+ix;
            tile[dy*8+sub_x][position]=rna((2*static_cast<int>(x[offset])-255)*512,510);
        }
    }
    __syncthreads();
    #pragma unroll
    for(int i=threadIdx.x;i<64*32;i+=256) {
        int channel=i/32,pos=i%32;
        if(ox+pos<ow)
            out[((static_cast<int64_t>(plane)*64+channel)*oh+oy)*ow+ox+pos]=tile[channel][pos];
    }
}
__global__ void bytes_unshuffle_yuv420_kernel(int16_t* out,const uint8_t* x,
    int h,int w,int oh,int ow,int tiles_w) {
    __shared__ int32_t tile[64][36];
    int ox=(blockIdx.x%tiles_w)*32,group=blockIdx.x/tiles_w;
    int oy=group%oh,plane=group/oh;
    int frame=plane/3,comp=plane%3;
    int position=threadIdx.x/8,sub_x=threadIdx.x%8;
    if(ox+position<ow) {
        int ix=ox*8+threadIdx.x;
        ix=ix<w?ix:w-1;
        const uint8_t* frame_ptr=x+static_cast<int64_t>(frame)*(h/2)*w*3;
        if(comp==0) {
            #pragma unroll
            for(int dy=0;dy<8;++dy) {
                int iy=oy*8+dy;
                iy=iy<h?iy:h-1;
                tile[dy*8+sub_x][position]=rna((2*static_cast<int>(frame_ptr[iy*w+ix])-255)*512,510);
            }
        } else {
            int uv_h=h>>1,uv_w=w>>1;
            const uint8_t* uv_ptr=frame_ptr+static_cast<int64_t>(h)*w+(comp==2?static_cast<int64_t>(uv_h)*uv_w:0);
            int uv_x=ix>>1;
            #pragma unroll
            for(int dy=0;dy<8;++dy) {
                int iy=oy*8+dy;
                iy=iy<h?iy:h-1;
                int uv_y=iy>>1;
                tile[dy*8+sub_x][position]=rna((2*static_cast<int>(uv_ptr[uv_y*uv_w+uv_x])-255)*512,510);
            }
        }
    }
    __syncthreads();
    #pragma unroll
    for(int i=threadIdx.x;i<64*32;i+=256) {
        int channel=i/32,pos=i%32;
        if(ox+pos<ow)
            out[((static_cast<int64_t>(plane)*64+channel)*oh+oy)*ow+ox+pos]=tile[channel][pos];
    }
}
__global__ void bytes_unshuffle_yuv420_nhwc_kernel(int16_t* __restrict__ out,const uint8_t* __restrict__ x,
    int h,int w,int oh,int ow,int tiles_w,int out_c) {
    __shared__ int16_t tile[32][66];
    int ox=(blockIdx.x%tiles_w)*32,group=blockIdx.x/tiles_w;
    int oy=group%oh,plane=group/oh;
    int frame=plane/3,comp=plane%3;
    int position=threadIdx.x/8,sub_x=threadIdx.x%8;
    if(ox+position<ow) {
        int ix=ox*8+threadIdx.x;
        ix=ix<w?ix:w-1;
        const uint8_t* frame_ptr=x+static_cast<int64_t>(frame)*(h/2)*w*3;
        if(comp==0) {
            #pragma unroll
            for(int dy=0;dy<8;++dy) {
                int iy=oy*8+dy;
                iy=iy<h?iy:h-1;
                tile[position][dy*8+sub_x]=rna((2*static_cast<int>(frame_ptr[iy*w+ix])-255)*512,510);
            }
        } else {
            int uv_h=h>>1,uv_w=w>>1;
            const uint8_t* uv_ptr=frame_ptr+static_cast<int64_t>(h)*w+(comp==2?static_cast<int64_t>(uv_h)*uv_w:0);
            int uv_x=ix>>1;
            #pragma unroll
            for(int dy=0;dy<8;++dy) {
                int iy=oy*8+dy;
                iy=iy<h?iy:h-1;
                int uv_y=iy>>1;
                tile[position][dy*8+sub_x]=rna((2*static_cast<int>(uv_ptr[uv_y*uv_w+uv_x])-255)*512,510);
            }
        }
    }
    __syncthreads();
    #pragma unroll
    for(int i=threadIdx.x;i<32*64;i+=256) {
        int pos=i/64,channel=i%64;
        if(ox+pos<ow)
            out[(static_cast<int64_t>(oy*ow+ox+pos)*out_c)+plane*64+channel]=tile[pos][channel];
    }
}
__global__ void cat_pack_nhwc_kernel(int16_t* __restrict__ out,
    const int16_t* __restrict__ a,const int16_t* __restrict__ b,
    int c1,int c2,int hw) {
    __shared__ int32_t tile[32][33];
    int c_total=c1+c2;
    int pos_base=blockIdx.x*32,c_base=blockIdx.y*32,batch=blockIdx.z;
    int tx=threadIdx.x,ty=threadIdx.y;
    int pos=pos_base+tx;
    #pragma unroll
    for(int dy=0;dy<32;dy+=8) {
        int c=c_base+ty+dy;
        if(pos<hw && c<c_total) {
            tile[ty+dy][tx]=(c<c1)
                ? a[(static_cast<int64_t>(batch)*c1+c)*hw+pos]
                : b[(static_cast<int64_t>(batch)*c2+(c-c1))*hw+pos];
        }
    }
    __syncthreads();
    int out_c=c_base+tx;
    #pragma unroll
    for(int dy=0;dy<32;dy+=8) {
        int out_pos=pos_base+ty+dy;
        if(out_pos<hw && out_c<c_total)
            out[(static_cast<int64_t>(batch)*hw+out_pos)*c_total+out_c]=tile[tx][ty+dy];
    }
}
__global__ void copy_nchw_to_nhwc_slice_kernel(int16_t* __restrict__ out,
    const int16_t* __restrict__ src,int src_c,int out_c,int c_offset,int hw) {
    __shared__ int32_t tile[32][33];
    int pos_base=blockIdx.x*32,c_base=blockIdx.y*32,batch=blockIdx.z;
    int tx=threadIdx.x,ty=threadIdx.y;
    int pos=pos_base+tx;
    #pragma unroll
    for(int dy=0;dy<32;dy+=8) {
        int c=c_base+ty+dy;
        if(pos<hw && c<src_c)
            tile[ty+dy][tx]=src[(static_cast<int64_t>(batch)*src_c+c)*hw+pos];
    }
    __syncthreads();
    int dst_c=c_base+tx;
    #pragma unroll
    for(int dy=0;dy<32;dy+=8) {
        int out_pos=pos_base+ty+dy;
        if(out_pos<hw && dst_c<src_c)
            out[(static_cast<int64_t>(batch)*hw+out_pos)*out_c+c_offset+dst_c]=tile[tx][ty+dy];
    }
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
struct RawFrames {
    const int16_t* data[8];
};
__device__ uint8_t raw_pixel(const int16_t* x,int i,int width,int height,
                           int input_width,int input_height) {
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
    return value<0?0:(value>255?255:value);
}
__global__ void raw_frame_kernel(uint8_t* out,const int16_t* x,int count,int width,int height,
    int input_width,int input_height) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) out[i]=raw_pixel(x,i,width,height,input_width,input_height);
}
__global__ void raw_frames_kernel(uint8_t* out,RawFrames frames,int frame_bytes,
    int width,int height,int input_width,int input_height) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<frame_bytes)
        out[static_cast<int64_t>(blockIdx.y)*frame_bytes+i]=
            raw_pixel(frames.data[blockIdx.y],i,width,height,input_width,input_height);
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

torch::Tensor uf_bytes_unshuffle(const torch::Tensor& input) {
    TORCH_CHECK(input.is_cuda() && input.scalar_type()==at::kByte &&
                (input.dim()==4 || input.dim()==5) &&
                input.numel()>0 && input.numel()<=INT32_MAX,"nonempty CUDA NCHW or YUV420 bytes required");
    c10::cuda::CUDAGuard guard(input.device()); auto x=input.contiguous();
    if(x.dim()==5) {
        TORCH_CHECK(x.size(2)==3 && x.size(4)%2==0,"invalid packed YUV420 dimensions");
        int64_t h=x.size(3)*2,w=x.size(4);
        int64_t planes=x.size(0)*x.size(1)*3;
        int64_t oh=(h+15)/16*2,ow=(w+15)/16*2;
        int64_t count=planes*64*oh*ow;
        TORCH_CHECK(count<=INT32_MAX,"unshuffled source exceeds indexing bounds");
        auto out=torch::empty({x.size(0),x.size(1)*3*64,oh,ow},x.options().dtype(at::kShort));
        int64_t tiles_w=(ow+31)/32,blocks=planes*oh*tiles_w;
        bytes_unshuffle_yuv420_kernel<<<blocks,256,0,c10::cuda::getCurrentCUDAStream()>>>(
            out.data_ptr<int16_t>(),x.data_ptr<uint8_t>(),h,w,oh,ow,tiles_w);
        C10_CUDA_KERNEL_LAUNCH_CHECK(); return out;
    }
    int64_t oh=(x.size(2)+15)/16*2,ow=(x.size(3)+15)/16*2;
    int64_t count=x.size(0)*x.size(1)*64*oh*ow;
    TORCH_CHECK(count<=INT32_MAX,"unshuffled source exceeds indexing bounds");
    auto out=torch::empty({x.size(0),x.size(1)*64,oh,ow},x.options().dtype(at::kShort));
    int64_t tiles_w=(ow+31)/32,blocks=x.size(0)*x.size(1)*oh*tiles_w;
    bytes_unshuffle_kernel<<<blocks,256,0,c10::cuda::getCurrentCUDAStream()>>>(
        out.data_ptr<int16_t>(),x.data_ptr<uint8_t>(),x.size(2),x.size(3),oh,ow,tiles_w);
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return out;
}
std::pair<torch::Tensor,torch::Tensor> uf_cat_pack_nhwc(
    const torch::Tensor& a_in,const torch::Tensor& b_in) {
    auto a=a_in.contiguous(),b=b_in.contiguous();
    check(a); check(b);
    TORCH_CHECK(a.device()==b.device() && a.dim()==4 && b.dim()==4 &&
                a.size(0)==b.size(0) && a.size(2)==b.size(2) && a.size(3)==b.size(3),
                "cat_pack_nhwc shape/device mismatch");
    c10::cuda::CUDAGuard guard(a.device());
    int64_t batch=a.size(0),c1=a.size(1),c2=b.size(1),h=a.size(2),w=a.size(3),hw=h*w;
    auto packed=torch::empty({batch,h,w,c1+c2},a.options());
    dim3 grid((hw+31)/32,(c1+c2+31)/32,batch);
    cat_pack_nhwc_kernel<<<grid,dim3(32,8),0,c10::cuda::getCurrentCUDAStream()>>>(
        packed.data_ptr<int16_t>(),a.data_ptr<int16_t>(),b.data_ptr<int16_t>(),c1,c2,hw);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {packed.permute({0,3,1,2}),packed};
}
std::pair<torch::Tensor,torch::Tensor> uf_bytes_unshuffle_cat_nhwc(
    const torch::Tensor& input,const torch::Tensor& ctx_in) {
    TORCH_CHECK(input.is_cuda() && input.scalar_type()==at::kByte && input.dim()==5 &&
                input.size(0)==1 && input.size(2)==3 && input.size(4)%2==0 &&
                input.numel()>0 && input.numel()<=INT32_MAX,
                "invalid packed YUV420 dimensions for unshuffle_cat_nhwc");
    auto x=input.contiguous(),ctx=ctx_in.contiguous();
    check(ctx);
    int64_t h=x.size(3)*2,w=x.size(4);
    int64_t planes=x.size(1)*3,src_c=planes*64;
    int64_t oh=(h+15)/16*2,ow=(w+15)/16*2,hw=oh*ow;
    TORCH_CHECK(ctx.device()==x.device() && ctx.dim()==4 && ctx.size(0)==1 &&
                ctx.size(2)==oh && ctx.size(3)==ow,"context shape mismatch in unshuffle_cat_nhwc");
    int64_t ctx_c=ctx.size(1),total_c=src_c+ctx_c;
    TORCH_CHECK(total_c*hw<=INT32_MAX,"unshuffled cat exceeds indexing bounds");
    c10::cuda::CUDAGuard guard(x.device());
    auto packed=torch::empty({1,oh,ow,total_c},ctx.options());
    auto stream=c10::cuda::getCurrentCUDAStream();
    int64_t tiles_w=(ow+31)/32,blocks=planes*oh*tiles_w;
    bytes_unshuffle_yuv420_nhwc_kernel<<<blocks,256,0,stream>>>(
        packed.data_ptr<int16_t>(),x.data_ptr<uint8_t>(),h,w,oh,ow,tiles_w,total_c);
    dim3 grid((hw+31)/32,(ctx_c+31)/32,1);
    copy_nchw_to_nhwc_slice_kernel<<<grid,dim3(32,8),0,stream>>>(
        packed.data_ptr<int16_t>(),ctx.data_ptr<int16_t>(),ctx_c,total_c,src_c,hw);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {packed.permute({0,3,1,2}),packed};
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

torch::Tensor uf_raw_frames(const std::vector<torch::Tensor>& inputs,int width,int height) {
    TORCH_CHECK(!inputs.empty() && inputs.size()<=8,"one to eight YUV420 frames required");
    if(inputs.size()==1) return uf_raw_frame(inputs[0],width,height).view({1,-1});
    const auto& first=inputs[0];
    TORCH_CHECK(first.is_cuda() && first.dim()==4 && first.size(0)==1 && first.size(1)==3 &&
                width>0 && height>0 && width%2==0 && height%2==0 &&
                width<=first.size(3) && height<=first.size(2),"invalid YUV420 dimensions");
    c10::cuda::CUDAGuard guard(first.device());
    int64_t frame_bytes=static_cast<int64_t>(width)*height*3/2;
    int64_t count=frame_bytes*inputs.size();
    TORCH_CHECK(count<=INT32_MAX,"YUV420 chunk exceeds indexing bounds");
    RawFrames pointers{};
    std::vector<torch::Tensor> contiguous;
    for(const auto& input:inputs) {
        TORCH_CHECK(input.device()==first.device() && input.sizes()==first.sizes(),
                    "YUV420 frame shape/device mismatch");
        contiguous.push_back(input.contiguous()); check(contiguous.back());
        pointers.data[contiguous.size()-1]=contiguous.back().data_ptr<int16_t>();
    }
    auto out=torch::empty({static_cast<int64_t>(inputs.size()),frame_bytes},first.options().dtype(at::kByte));
    raw_frames_kernel<<<dim3((frame_bytes+255)/256,inputs.size()),256,0,c10::cuda::getCurrentCUDAStream()>>>(
        out.data_ptr<uint8_t>(),pointers,frame_bytes,width,height,first.size(3),first.size(2));
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
