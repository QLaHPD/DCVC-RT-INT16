// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>

namespace {

template <typename scalar_t>
using Packed4DTensorAccessor32 = torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits>;
template <typename scalar_t>
using Packed1DTensorAccessor32 = torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits>;

constexpr int WEIGHT_SCALE = 8192;
constexpr int FEATURE_SCALE = 512;
constexpr int INT16_MIN_VAL = -32768;
constexpr int INT16_MAX_VAL = 32767;
constexpr int THREADS = 256;

__forceinline__ __device__ int32_t round_divide_int(const int32_t value, const int32_t denominator)
{
    const int32_t abs_value = value >= 0 ? value : -value;
    const int32_t rounded = (abs_value + denominator / 2) / denominator;
    return value >= 0 ? rounded : -rounded;
}

__forceinline__ __device__ int16_t clip_to_int16(const int32_t value)
{
    const int32_t clipped = max(INT16_MIN_VAL, min(INT16_MAX_VAL, value));
    return static_cast<int16_t>(clipped);
}

#include "int16_mma.cuh"

__forceinline__ __device__ int64_t broadcast_offset_4d(const int64_t linear_index,
                                                       const int B, const int C, const int H, const int W,
                                                       const int sB, const int sC, const int sH, const int sW)
{
    int64_t tmp = linear_index;
    const int w = tmp % W;
    tmp /= W;
    const int h = tmp % H;
    tmp /= H;
    const int c = tmp % C;
    const int b = tmp / C;
    const int sb = sB == 1 ? 0 : b;
    const int sc = sC == 1 ? 0 : c;
    const int sh = sH == 1 ? 0 : h;
    const int sw = sW == 1 ? 0 : w;
    return (((static_cast<int64_t>(sb) * sC + sc) * sH + sh) * sW + sw);
}

template <bool with_bias>
__global__ void conv2d_int16_kernel(Packed4DTensorAccessor32<int16_t> out,
                                    const Packed4DTensorAccessor32<int16_t> x,
                                    const Packed4DTensorAccessor32<int16_t> weight,
                                    const Packed1DTensorAccessor32<int16_t> bias,
                                    const int stride_h, const int stride_w,
                                    const int pad_h, const int pad_w, const int groups)
{
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int bz = blockIdx.z;
    const int batch = bz / out.size(1);
    const int out_channel = bz % out.size(1);

    if (batch >= out.size(0) || out_channel >= out.size(1) || out_y >= out.size(2) ||
        out_x >= out.size(3)) {
        return;
    }

    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    const int out_channels_per_group = out.size(1) / groups;
    const int in_channels_per_group = x.size(1) / groups;
    const int group = out_channel / out_channels_per_group;
    const int in_channel_begin = group * in_channels_per_group;

    int32_t sum = 0;
    for (int in_channel = 0; in_channel < in_channels_per_group; ++in_channel) {
        for (int ky = 0; ky < kernel_h; ++ky) {
            const int in_y = out_y * stride_h + ky - pad_h;
            if (in_y < 0 || in_y >= x.size(2)) {
                continue;
            }
            for (int kx = 0; kx < kernel_w; ++kx) {
                const int in_x = out_x * stride_w + kx - pad_w;
                if (in_x < 0 || in_x >= x.size(3)) {
                    continue;
                }
                const int32_t x_val = static_cast<int32_t>(x[batch][in_channel_begin + in_channel][in_y][in_x]);
                const int32_t w_val = static_cast<int32_t>(weight[out_channel][in_channel][ky][kx]);
                sum += x_val * w_val;
            }
        }
    }

    int32_t out_val = round_divide_int(sum, WEIGHT_SCALE);
    if constexpr (with_bias) {
        out_val += static_cast<int32_t>(bias[out_channel]);
    }
    out[batch][out_channel][out_y][out_x] = clip_to_int16(out_val);
}

template <bool with_bias>
__global__ void conv2d_int16_1x1_kernel(int16_t* out, const int16_t* x, const int16_t* weight,
                                        const int16_t* bias, const int64_t N,
                                        const int out_C, const int out_H, const int out_W,
                                        const int in_C, const int in_H, const int in_W,
                                        const int stride_h, const int stride_w,
                                        const int pad_h, const int pad_w, const int groups)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }

    int64_t tmp = idx;
    const int out_x = tmp % out_W;
    tmp /= out_W;
    const int out_y = tmp % out_H;
    tmp /= out_H;
    const int out_channel = tmp % out_C;
    const int batch = tmp / out_C;

    const int out_channels_per_group = out_C / groups;
    const int in_channels_per_group = in_C / groups;
    const int group = out_channel / out_channels_per_group;
    const int in_channel_begin = group * in_channels_per_group;
    const int in_y = out_y * stride_h - pad_h;
    const int in_x = out_x * stride_w - pad_w;

    int32_t sum = 0;
    if (in_y >= 0 && in_y < in_H && in_x >= 0 && in_x < in_W) {
        const int64_t x_base = (((static_cast<int64_t>(batch) * in_C + in_channel_begin) * in_H + in_y) * in_W + in_x);
        const int64_t weight_base = static_cast<int64_t>(out_channel) * in_channels_per_group;
        const int64_t x_channel_stride = static_cast<int64_t>(in_H) * in_W;
        for (int in_channel = 0; in_channel < in_channels_per_group; ++in_channel) {
            sum += static_cast<int32_t>(x[x_base + static_cast<int64_t>(in_channel) * x_channel_stride]) *
                   static_cast<int32_t>(weight[weight_base + in_channel]);
        }
    }

    int32_t out_val = round_divide_int(sum, WEIGHT_SCALE);
    if constexpr (with_bias) {
        out_val += static_cast<int32_t>(bias[out_channel]);
    }
    out[idx] = clip_to_int16(out_val);
}

// Treat a dense, stride-1 1x1 convolution as W[M, K] * X[K, N].  A warp owns
// several output channels and 64 spatial positions.  Keeping K as the innermost
// loop preserves the reference kernel's accumulation order for every output,
// while shared-memory tiles avoid re-reading the same activations for each
// output channel.
template <bool with_bias, int TILE_M, int ROWS_PER_THREAD>
__global__ __launch_bounds__(256) void conv2d_int16_1x1_tiled_kernel(
    int16_t* __restrict__ out,
    const int16_t* __restrict__ x,
    const int16_t* __restrict__ weight,
    const int16_t* __restrict__ bias,
    const int M, const int N, const int K)
{
    constexpr int TILE_N = 64;
    constexpr int TILE_K = 16;
    constexpr int COLS_PER_THREAD = 2;
    static_assert(TILE_M == 8 * ROWS_PER_THREAD);

    __shared__ int16_t weight_tile[TILE_M][TILE_K];
    __shared__ __align__(4) int16_t input_tile[TILE_K][TILE_N];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row_base = blockIdx.y * TILE_M + warp * ROWS_PER_THREAD;
    const int col_base = blockIdx.x * TILE_N + lane * COLS_PER_THREAD;
    const int batch = blockIdx.z;

    // Unsigned accumulation makes the existing low-32-bit wrap behavior
    // explicit.  Converting back to int32 before rounding matches the current
    // CUDA kernel, including inputs whose mathematical sum exceeds int32.
    uint32_t sums[ROWS_PER_THREAD][COLS_PER_THREAD] = {};

    for (int k_base = 0; k_base < K; k_base += TILE_K) {
        for (int linear = tid; linear < TILE_M * TILE_K; linear += blockDim.x) {
            const int tile_row = linear / TILE_K;
            const int tile_k = linear % TILE_K;
            const int row = blockIdx.y * TILE_M + tile_row;
            const int k = k_base + tile_k;
            weight_tile[tile_row][tile_k] =
                (row < M && k < K) ? weight[static_cast<int64_t>(row) * K + k] : 0;
        }
        for (int linear = tid; linear < TILE_K * TILE_N; linear += blockDim.x) {
            const int tile_k = linear / TILE_N;
            const int tile_col = linear % TILE_N;
            const int k = k_base + tile_k;
            const int col = blockIdx.x * TILE_N + tile_col;
            input_tile[tile_k][tile_col] =
                (k < K && col < N) ? x[(static_cast<int64_t>(batch) * K + k) * N + col] : 0;
        }
        __syncthreads();

        #pragma unroll
        for (int tile_k = 0; tile_k < TILE_K; ++tile_k) {
            const int packed_x = *reinterpret_cast<const int32_t*>(&input_tile[tile_k][lane * 2]);
            const int32_t x0 = static_cast<int16_t>(packed_x & 0xFFFF);
            const int32_t x1 = static_cast<int16_t>(static_cast<uint32_t>(packed_x) >> 16);

            #pragma unroll
            for (int r = 0; r < ROWS_PER_THREAD; ++r) {
                const int32_t w = static_cast<int32_t>(weight_tile[warp * ROWS_PER_THREAD + r][tile_k]);
                sums[r][0] += static_cast<uint32_t>(w * x0);
                sums[r][1] += static_cast<uint32_t>(w * x1);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const int row = row_base + r;
        if (row >= M) {
            continue;
        }
        const int32_t bias_value = with_bias ? static_cast<int32_t>(bias[row]) : 0;
        #pragma unroll
        for (int c = 0; c < COLS_PER_THREAD; ++c) {
            const int col = col_base + c;
            if (col >= N) {
                continue;
            }
            int32_t out_val = round_divide_int(static_cast<int32_t>(sums[r][c]), WEIGHT_SCALE);
            out_val += bias_value;
            out[(static_cast<int64_t>(batch) * M + row) * N + col] = clip_to_int16(out_val);
        }
    }
}

// Dense stride-1 3x3 convolution as an implicit GEMM.  Flattening K as
// [input_channel, kernel_y, kernel_x] exactly matches the reference loop order;
// out-of-image samples are loaded as zero instead of materializing im2col.
template <bool with_bias>
__global__ __launch_bounds__(128) void conv2d_int16_3x3_tiled_kernel(
    int16_t* __restrict__ out,
    const int16_t* __restrict__ x,
    const int16_t* __restrict__ weight,
    const int16_t* __restrict__ bias,
    const int M, const int N, const int in_C,
    const int in_H, const int in_W, const int out_W)
{
    constexpr int TILE_M = 32;
    constexpr int TILE_N = 32;
    constexpr int TILE_K = 16;
    constexpr int ROWS_PER_THREAD = 8;

    __shared__ int16_t weight_tile[TILE_M][TILE_K];
    __shared__ int16_t input_tile[TILE_K][TILE_N];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row_base = blockIdx.y * TILE_M + warp * ROWS_PER_THREAD;
    const int col = blockIdx.x * TILE_N + lane;
    const int batch = blockIdx.z;
    const int K = in_C * 9;
    uint32_t sums[ROWS_PER_THREAD] = {};

    for (int k_base = 0; k_base < K; k_base += TILE_K) {
        for (int linear = tid; linear < TILE_M * TILE_K; linear += blockDim.x) {
            const int tile_row = linear / TILE_K;
            const int tile_k = linear % TILE_K;
            const int row = blockIdx.y * TILE_M + tile_row;
            const int k = k_base + tile_k;
            weight_tile[tile_row][tile_k] =
                (row < M && k < K) ? weight[static_cast<int64_t>(row) * K + k] : 0;
        }
        for (int linear = tid; linear < TILE_K * TILE_N; linear += blockDim.x) {
            const int tile_k = linear / TILE_N;
            const int tile_col = linear % TILE_N;
            const int k = k_base + tile_k;
            const int out_col = blockIdx.x * TILE_N + tile_col;
            int16_t value = 0;
            if (k < K && out_col < N) {
                const int in_channel = k / 9;
                const int kernel_offset = k - in_channel * 9;
                const int kernel_y = kernel_offset / 3;
                const int kernel_x = kernel_offset - kernel_y * 3;
                const int out_y = out_col / out_W;
                const int out_x = out_col - out_y * out_W;
                const int in_y = out_y + kernel_y - 1;
                const int in_x = out_x + kernel_x - 1;
                if (in_y >= 0 && in_y < in_H && in_x >= 0 && in_x < in_W) {
                    const int64_t input_index =
                        ((static_cast<int64_t>(batch) * in_C + in_channel) * in_H + in_y) * in_W + in_x;
                    value = x[input_index];
                }
            }
            input_tile[tile_k][tile_col] = value;
        }
        __syncthreads();

        #pragma unroll
        for (int tile_k = 0; tile_k < TILE_K; ++tile_k) {
            const int32_t x_value = static_cast<int32_t>(input_tile[tile_k][lane]);
            #pragma unroll
            for (int r = 0; r < ROWS_PER_THREAD; ++r) {
                const int32_t w = static_cast<int32_t>(weight_tile[warp * ROWS_PER_THREAD + r][tile_k]);
                sums[r] += static_cast<uint32_t>(w * x_value);
            }
        }
        __syncthreads();
    }

    if (col >= N) {
        return;
    }
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const int row = row_base + r;
        if (row >= M) {
            continue;
        }
        int32_t out_val = round_divide_int(static_cast<int32_t>(sums[r]), WEIGHT_SCALE);
        if constexpr (with_bias) {
            out_val += static_cast<int32_t>(bias[row]);
        }
        out[(static_cast<int64_t>(batch) * M + row) * N + col] = clip_to_int16(out_val);
    }
}

template <bool with_bias>
__global__ void conv2d_int16_depthwise_3x3_kernel(int16_t* out, const int16_t* x, const int16_t* weight,
                                                   const int16_t* bias, const int64_t N,
                                                   const int C, const int out_H, const int out_W,
                                                   const int in_H, const int in_W)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }

    int64_t tmp = idx;
    const int out_x = tmp % out_W;
    tmp /= out_W;
    const int out_y = tmp % out_H;
    tmp /= out_H;
    const int channel = tmp % C;
    const int batch = tmp / C;

    const int64_t x_base = ((static_cast<int64_t>(batch) * C + channel) * in_H) * in_W;
    const int64_t w_base = static_cast<int64_t>(channel) * 9;
    int32_t sum = 0;

    #pragma unroll
    for (int ky = 0; ky < 3; ++ky) {
        const int in_y = out_y + ky - 1;
        if (in_y < 0 || in_y >= in_H) {
            continue;
        }
        #pragma unroll
        for (int kx = 0; kx < 3; ++kx) {
            const int in_x = out_x + kx - 1;
            if (in_x < 0 || in_x >= in_W) {
                continue;
            }
            sum += static_cast<int32_t>(x[x_base + static_cast<int64_t>(in_y) * in_W + in_x]) *
                   static_cast<int32_t>(weight[w_base + ky * 3 + kx]);
        }
    }

    int32_t out_val = round_divide_int(sum, WEIGHT_SCALE);
    if constexpr (with_bias) {
        out_val += static_cast<int32_t>(bias[channel]);
    }
    out[idx] = clip_to_int16(out_val);
}

template <bool with_bias>
__global__ void conv2d_int16_stride2_2x2_kernel(int16_t* out, const int16_t* x, const int16_t* weight,
                                                 const int16_t* bias, const int64_t N,
                                                 const int out_C, const int out_H, const int out_W,
                                                 const int in_C, const int in_H, const int in_W)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }

    int64_t tmp = idx;
    const int out_x = tmp % out_W;
    tmp /= out_W;
    const int out_y = tmp % out_H;
    tmp /= out_H;
    const int out_channel = tmp % out_C;
    const int batch = tmp / out_C;

    const int in_y = out_y * 2;
    const int in_x = out_x * 2;
    const int64_t x_batch_base = static_cast<int64_t>(batch) * in_C * in_H * in_W;
    const int64_t w_out_base = static_cast<int64_t>(out_channel) * in_C * 4;

    int32_t sum = 0;
    for (int in_channel = 0; in_channel < in_C; ++in_channel) {
        const int64_t x_base = x_batch_base + static_cast<int64_t>(in_channel) * in_H * in_W +
                               static_cast<int64_t>(in_y) * in_W + in_x;
        const int64_t w_base = w_out_base + static_cast<int64_t>(in_channel) * 4;
        sum += static_cast<int32_t>(x[x_base]) * static_cast<int32_t>(weight[w_base]);
        sum += static_cast<int32_t>(x[x_base + 1]) * static_cast<int32_t>(weight[w_base + 1]);
        sum += static_cast<int32_t>(x[x_base + in_W]) * static_cast<int32_t>(weight[w_base + 2]);
        sum += static_cast<int32_t>(x[x_base + in_W + 1]) * static_cast<int32_t>(weight[w_base + 3]);
    }

    int32_t out_val = round_divide_int(sum, WEIGHT_SCALE);
    if constexpr (with_bias) {
        out_val += static_cast<int32_t>(bias[out_channel]);
    }
    out[idx] = clip_to_int16(out_val);
}

template <bool with_bias>
__global__ void conv2d_int16_stride2_3x3_kernel(int16_t* out, const int16_t* x, const int16_t* weight,
                                                 const int16_t* bias, const int64_t N,
                                                 const int out_C, const int out_H, const int out_W,
                                                 const int in_C, const int in_H, const int in_W)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }

    int64_t tmp = idx;
    const int out_x = tmp % out_W;
    tmp /= out_W;
    const int out_y = tmp % out_H;
    tmp /= out_H;
    const int out_channel = tmp % out_C;
    const int batch = tmp / out_C;

    const int in_y_center = out_y * 2;
    const int in_x_center = out_x * 2;
    const int64_t x_batch_base = static_cast<int64_t>(batch) * in_C * in_H * in_W;
    const int64_t w_out_base = static_cast<int64_t>(out_channel) * in_C * 9;

    int32_t sum = 0;
    for (int in_channel = 0; in_channel < in_C; ++in_channel) {
        const int64_t x_channel_base = x_batch_base + static_cast<int64_t>(in_channel) * in_H * in_W;
        const int64_t w_base = w_out_base + static_cast<int64_t>(in_channel) * 9;
        #pragma unroll
        for (int ky = 0; ky < 3; ++ky) {
            const int in_y = in_y_center + ky - 1;
            if (in_y < 0 || in_y >= in_H) {
                continue;
            }
            #pragma unroll
            for (int kx = 0; kx < 3; ++kx) {
                const int in_x = in_x_center + kx - 1;
                if (in_x < 0 || in_x >= in_W) {
                    continue;
                }
                sum += static_cast<int32_t>(x[x_channel_base + static_cast<int64_t>(in_y) * in_W + in_x]) *
                       static_cast<int32_t>(weight[w_base + ky * 3 + kx]);
            }
        }
    }

    int32_t out_val = round_divide_int(sum, WEIGHT_SCALE);
    if constexpr (with_bias) {
        out_val += static_cast<int32_t>(bias[out_channel]);
    }
    out[idx] = clip_to_int16(out_val);
}

__global__ void wsilu_chunk_add_int16_kernel(int16_t* out, const int16_t* x, const int16_t* lut,
                                             const int64_t out_N, const int out_C, const int H, const int W)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= out_N) {
        return;
    }

    int64_t tmp = idx;
    const int w = tmp % W;
    tmp /= W;
    const int h = tmp % H;
    tmp /= H;
    const int c = tmp % out_C;
    const int b = tmp / out_C;

    const int64_t in_idx0 = (((static_cast<int64_t>(b) * (out_C * 2) + c) * H + h) * W + w);
    const int64_t in_idx1 = in_idx0 + static_cast<int64_t>(out_C) * H * W;
    const int32_t v0 = static_cast<int32_t>(lut[static_cast<int32_t>(x[in_idx0]) + 32768]);
    const int32_t v1 = static_cast<int32_t>(lut[static_cast<int32_t>(x[in_idx1]) + 32768]);
    out[idx] = clip_to_int16(v0 + v1);
}

__global__ void bias_wsilu_depthwise_conv2d_int16_kernel(int16_t* out, const int16_t* x,
                                                         const int16_t* weight, const int16_t* bias,
                                                         const int16_t* lut, const int64_t N,
                                                         const int C, const int H, const int W)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }

    int64_t tmp = idx;
    const int w = tmp % W;
    tmp /= W;
    const int h = tmp % H;
    tmp /= H;
    const int c = tmp % C;
    const int b = tmp / C;

    const int64_t x_base = ((static_cast<int64_t>(b) * C + c) * H) * W;
    const int64_t w_base = static_cast<int64_t>(c) * 9;
    int32_t sum = 0;

    #pragma unroll
    for (int ky = 0; ky < 3; ++ky) {
        const int in_y = h + ky - 1;
        if (in_y < 0 || in_y >= H) {
            continue;
        }
        #pragma unroll
        for (int kx = 0; kx < 3; ++kx) {
            const int in_x = w + kx - 1;
            if (in_x < 0 || in_x >= W) {
                continue;
            }
            const int16_t x_val = x[x_base + static_cast<int64_t>(in_y) * W + in_x];
            const int16_t act = lut[static_cast<int32_t>(x_val) + 32768];
            sum += static_cast<int32_t>(act) * static_cast<int32_t>(weight[w_base + ky * 3 + kx]);
        }
    }

    const int32_t out_val = round_divide_int(sum, WEIGHT_SCALE) + static_cast<int32_t>(bias[c]);
    out[idx] = clip_to_int16(out_val);
}

__global__ void add_bias_int16_kernel(int16_t* out, const int16_t* x, const int16_t* bias,
                                      const int64_t N, const int C, const int HW)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }
    const int c = (idx / HW) % C;
    const int32_t val = static_cast<int32_t>(x[idx]) + static_cast<int32_t>(bias[c]);
    out[idx] = clip_to_int16(val);
}

__global__ void add_tensors_int16_kernel(int16_t* out, const int16_t* x, const int16_t* y, const int64_t N)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }
    out[idx] = clip_to_int16(static_cast<int32_t>(x[idx]) + static_cast<int32_t>(y[idx]));
}

__global__ void mul_feature_scale_int16_kernel(int16_t* out, const int16_t* x, const int16_t* scale,
                                               const int64_t N,
                                               const int B, const int C, const int H, const int W,
                                               const int sB, const int sC, const int sH, const int sW)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }
    const int64_t scale_idx = broadcast_offset_4d(idx, B, C, H, W, sB, sC, sH, sW);
    const int32_t val = static_cast<int32_t>(x[idx]) * static_cast<int32_t>(scale[scale_idx]);
    out[idx] = clip_to_int16(round_divide_int(val, FEATURE_SCALE));
}

__global__ void reciprocal_scale_int16_kernel(int16_t* out_q_dec, int16_t* out_recip, const int16_t* q_dec,
                                              const int64_t N, const int32_t min_q)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }
    const int32_t clamped = max(static_cast<int32_t>(q_dec[idx]), min_q);
    out_q_dec[idx] = static_cast<int16_t>(clamped);
    out_recip[idx] = clip_to_int16(round_divide_int(FEATURE_SCALE * FEATURE_SCALE, clamped));
}

__global__ void apply_lut_int16_kernel(int16_t* out, const int16_t* x, const int16_t* lut, const int64_t N)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }
    const int64_t lut_idx = static_cast<int64_t>(static_cast<int32_t>(x[idx]) + 32768);
    out[idx] = lut[lut_idx];
}

__global__ void process_with_mask_int16_kernel(int16_t* y_res, int16_t* y_q, int16_t* y_hat, int16_t* s_hat,
                                               const int16_t* y, const int16_t* scales,
                                               const int16_t* means, const int16_t* mask,
                                               const int64_t N, const int32_t force_zero_thres)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }

    if (mask[idx] == 0) {
        y_res[idx] = 0;
        y_q[idx] = 0;
        y_hat[idx] = 0;
        s_hat[idx] = 0;
        return;
    }

    const int16_t scale_val = scales[idx];
    const int16_t mean_val = means[idx];
    const int32_t residual = static_cast<int32_t>(y[idx]) - static_cast<int32_t>(mean_val);
    int32_t q_val = round_divide_int(residual, FEATURE_SCALE);
    q_val = max(-128, min(127, q_val));
    if (force_zero_thres >= 0 && static_cast<int32_t>(scale_val) <= force_zero_thres) {
        q_val = 0;
    }

    y_res[idx] = clip_to_int16(residual);
    y_q[idx] = static_cast<int16_t>(q_val);
    y_hat[idx] = clip_to_int16(q_val * FEATURE_SCALE + static_cast<int32_t>(mean_val));
    s_hat[idx] = scale_val;
}

__global__ void combine_for_reading_int16_kernel(int16_t* out, const int16_t* x, const int16_t* mask,
                                                 const int64_t out_N, const int out_C, const int H, const int W,
                                                 const int parts)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= out_N) {
        return;
    }

    int64_t tmp = idx;
    const int w = tmp % W;
    tmp /= W;
    const int h = tmp % H;
    tmp /= H;
    const int c = tmp % out_C;
    const int b = tmp / out_C;

    int32_t sum = 0;
    for (int part = 0; part < parts; ++part) {
        const int in_c = c + part * out_C;
        const int64_t in_idx = (((static_cast<int64_t>(b) * (out_C * parts) + in_c) * H + h) * W + w);
        if (mask[in_idx] != 0) {
            sum += static_cast<int32_t>(x[in_idx]);
        }
    }
    out[idx] = clip_to_int16(sum);
}

__global__ void restore_y_parts_int16_kernel(int16_t* out, const int16_t* y, const int16_t* means,
                                             const int16_t* mask, const int64_t out_N,
                                             const int out_C, const int y_C, const int H, const int W)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= out_N) {
        return;
    }

    if (mask[idx] == 0) {
        out[idx] = 0;
        return;
    }

    int64_t tmp = idx;
    const int w = tmp % W;
    tmp /= W;
    const int h = tmp % H;
    tmp /= H;
    const int c = tmp % out_C;
    const int b = tmp / out_C;
    const int src_c = c % y_C;
    const int64_t y_idx = (((static_cast<int64_t>(b) * y_C + src_c) * H + h) * W + w);
    out[idx] = clip_to_int16(static_cast<int32_t>(y[y_idx]) * FEATURE_SCALE +
                             static_cast<int32_t>(means[idx]));
}

__global__ void build_index_dec_int16_kernel(uint8_t* out, bool* skip_cond, const int16_t* scales,
                                             const uint8_t* lut, const int64_t N,
                                             const int32_t skip_thres, const bool with_skip)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }
    const int16_t scale_val = scales[idx];
    out[idx] = lut[static_cast<int32_t>(scale_val) + 32768];
    if (with_skip) {
        skip_cond[idx] = static_cast<int32_t>(scale_val) > skip_thres;
    }
}

__global__ void build_index_enc_int16_kernel(int16_t* out, bool* skip_cond, const int16_t* symbols,
                                             const int16_t* scales, const uint8_t* lut,
                                             const int64_t N, const int32_t skip_thres,
                                             const bool with_skip)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }
    const int16_t scale_val = scales[idx];
    const int32_t packed = (static_cast<int32_t>(symbols[idx]) << 8) +
                           static_cast<int32_t>(lut[static_cast<int32_t>(scale_val) + 32768]);
    out[idx] = static_cast<int16_t>(packed);
    if (with_skip) {
        skip_cond[idx] = static_cast<int32_t>(scale_val) > skip_thres;
    }
}

__global__ void add_and_multiply_int16_kernel(int16_t* out, const int16_t* x0, const int16_t* x1,
                                              const int16_t* q, const int64_t N,
                                              const int B, const int C, const int H, const int W,
                                              const int qB, const int qC, const int qH, const int qW)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }
    const int64_t q_idx = broadcast_offset_4d(idx, B, C, H, W, qB, qC, qH, qW);
    const int32_t sum = static_cast<int32_t>(x0[idx]) + static_cast<int32_t>(x1[idx]);
    out[idx] = clip_to_int16(round_divide_int(sum * static_cast<int32_t>(q[q_idx]), FEATURE_SCALE));
}

__global__ void bias_quant_int16_kernel(int16_t* out, const int16_t* x, const int16_t* bias,
                                        const int16_t* q, const int64_t N,
                                        const int C, const int HW,
                                        const int B, const int H, const int W,
                                        const int qB, const int qC, const int qH, const int qW)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }
    const int c = (idx / HW) % C;
    const int64_t q_idx = broadcast_offset_4d(idx, B, C, H, W, qB, qC, qH, qW);
    const int32_t biased = static_cast<int32_t>(x[idx]) + static_cast<int32_t>(bias[c]);
    out[idx] = clip_to_int16(round_divide_int(biased * static_cast<int32_t>(q[q_idx]), FEATURE_SCALE));
}

__global__ void bias_pixel_shuffle_2_int16_kernel(int16_t* out, const int16_t* x, const int16_t* bias,
                                                  const int64_t out_N, const int B, const int out_C,
                                                  const int out_H, const int out_W,
                                                  const int in_H, const int in_W)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= out_N) {
        return;
    }

    int64_t tmp = idx;
    const int w = tmp % out_W;
    tmp /= out_W;
    const int h = tmp % out_H;
    tmp /= out_H;
    const int c = tmp % out_C;
    const int b = tmp / out_C;

    const int in_h = h / 2;
    const int in_w = w / 2;
    const int sub_h = h % 2;
    const int sub_w = w % 2;
    const int in_c = c * 4 + sub_h * 2 + sub_w;
    const int64_t in_idx = (((static_cast<int64_t>(b) * (out_C * 4) + in_c) * in_H + in_h) * in_W + in_w);
    const int32_t val = static_cast<int32_t>(x[in_idx]) + static_cast<int32_t>(bias[in_c]);
    out[idx] = clip_to_int16(val);
}

__global__ void bias_pixel_shuffle_8_int16_kernel(int16_t* out, const int16_t* x, const int16_t* bias,
                                                  const int64_t out_N, const int B, const int out_C,
                                                  const int out_H, const int out_W,
                                                  const int in_H, const int in_W)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= out_N) {
        return;
    }

    int64_t tmp = idx;
    const int w = tmp % out_W;
    tmp /= out_W;
    const int h = tmp % out_H;
    tmp /= out_H;
    const int c = tmp % out_C;
    const int b = tmp / out_C;

    const int in_h = h / 8;
    const int in_w = w / 8;
    const int sub_h = h % 8;
    const int sub_w = w % 8;
    const int in_c = c * 64 + sub_h * 8 + sub_w;
    const int64_t in_idx = (((static_cast<int64_t>(b) * (out_C * 64) + in_c) * in_H + in_h) * in_W + in_w);
    const int32_t val = static_cast<int32_t>(x[in_idx]) + static_cast<int32_t>(bias[in_c]);
    out[idx] = static_cast<int16_t>(max(0, min(FEATURE_SCALE, val)));
}

inline int64_t ceil_div_int64(const int64_t a, const int64_t b)
{
    return (a + b - 1) / b;
}

void check_cuda_int16(const torch::Tensor& x, const char* name)
{
    TORCH_CHECK(x.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kInt16, name, " must be int16");
}

void check_cuda_bool_or_empty(const torch::Tensor& x, const char* name)
{
    TORCH_CHECK(!x.defined() || x.scalar_type() == torch::kBool, name, " must be bool");
}

void check_same_shape(const torch::Tensor& x, const torch::Tensor& y, const char* x_name, const char* y_name)
{
    TORCH_CHECK(x.sizes() == y.sizes(), x_name, " and ", y_name, " must have the same shape");
}

void check_4d(const torch::Tensor& x, const char* name)
{
    TORCH_CHECK(x.dim() == 4, name, " must be 4D");
}

void check_broadcastable_4d(const torch::Tensor& x, const torch::Tensor& scale,
                            const char* x_name, const char* scale_name)
{
    check_4d(x, x_name);
    check_4d(scale, scale_name);
    for (int i = 0; i < 4; ++i) {
        TORCH_CHECK(scale.size(i) == 1 || scale.size(i) == x.size(i),
                    scale_name, " must broadcast to ", x_name);
    }
}

}  // namespace

torch::Tensor conv2d_int16_cuda(const torch::Tensor& x, const torch::Tensor& weight,
                                const torch::optional<torch::Tensor>& bias,
                                const int stride_h, const int stride_w,
                                const int pad_h, const int pad_w, const int groups)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kInt16, "x must be int16");
    TORCH_CHECK(weight.scalar_type() == torch::kInt16, "weight must be int16");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D");
    TORCH_CHECK(groups > 0, "groups must be positive");
    TORCH_CHECK(x.size(1) % groups == 0, "input channels must be divisible by groups");
    TORCH_CHECK(weight.size(0) % groups == 0, "output channels must be divisible by groups");
    TORCH_CHECK(weight.size(1) * groups == x.size(1), "weight shape is incompatible with groups");
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.value().scalar_type() == torch::kInt16, "bias must be int16");
        TORCH_CHECK(bias.value().dim() == 1, "bias must be 1D");
        TORCH_CHECK(bias.value().size(0) == weight.size(0), "bias size must equal output channels");
    }

    const auto x_contiguous = x.contiguous();
    const auto weight_contiguous = weight.contiguous();
    const int out_h = (x_contiguous.size(2) + 2 * pad_h - weight_contiguous.size(2)) / stride_h + 1;
    const int out_w = (x_contiguous.size(3) + 2 * pad_w - weight_contiguous.size(3)) / stride_w + 1;
    auto out = torch::empty({ x_contiguous.size(0), weight_contiguous.size(0), out_h, out_w },
                            x_contiguous.options());

    auto stream = c10::cuda::getCurrentCUDAStream();
    const int64_t out_numel = out.numel();
    const bool is_1x1 = weight_contiguous.size(2) == 1 && weight_contiguous.size(3) == 1;
    const bool is_tiled_1x1 = is_1x1 && groups == 1 && stride_h == 1 && stride_w == 1 &&
                              pad_h == 0 && pad_w == 0 && out_h == x_contiguous.size(2) &&
                              out_w == x_contiguous.size(3);
    const bool is_tiled_3x3 = weight_contiguous.size(2) == 3 && weight_contiguous.size(3) == 3 &&
                              groups == 1 && stride_h == 1 && stride_w == 1 &&
                              pad_h == 1 && pad_w == 1 && out_h == x_contiguous.size(2) &&
                              out_w == x_contiguous.size(3);
    const bool is_depthwise_3x3 = weight_contiguous.size(2) == 3 && weight_contiguous.size(3) == 3 &&
                                  groups == x_contiguous.size(1) && groups == weight_contiguous.size(0) &&
                                  weight_contiguous.size(1) == 1 &&
                                  stride_h == 1 && stride_w == 1 && pad_h == 1 && pad_w == 1;
    const bool is_stride2_2x2 = groups == 1 && weight_contiguous.size(2) == 2 && weight_contiguous.size(3) == 2 &&
                                stride_h == 2 && stride_w == 2 && pad_h == 0 && pad_w == 0;
    const bool is_stride2_3x3 = groups == 1 && weight_contiguous.size(2) == 3 && weight_contiguous.size(3) == 3 &&
                                stride_h == 2 && stride_w == 2 && pad_h == 1 && pad_w == 1;

    if ((is_tiled_1x1 || is_tiled_3x3) && int16_mma_supported(x.get_device())) {
        const auto bias_contiguous = bias.has_value() ? bias.value().contiguous() : torch::Tensor();
        const int16_t* bias_ptr = bias.has_value() ? bias_contiguous.data_ptr<int16_t>() : nullptr;
        const dim3 grid(ceil_div_int64(out_h * out_w, 32), ceil_div_int64(out.size(1), 64), out.size(0));
        const int K = x_contiguous.size(1) * (is_tiled_3x3 ? 9 : 1);
#define DCVC_LAUNCH_MMA(BIAS, CONV3) \
        conv2d_int16_mma_kernel<BIAS, CONV3><<<grid, 128, 0, stream>>>( \
            out.data_ptr<int16_t>(), x_contiguous.data_ptr<int16_t>(), weight_contiguous.data_ptr<int16_t>(), \
            bias_ptr, out.size(1), out_h*out_w, K, out_h, out_w)
        if (is_tiled_3x3) {
            if (bias.has_value()) { DCVC_LAUNCH_MMA(true, true); }
            else { DCVC_LAUNCH_MMA(false, true); }
        } else {
            if (bias.has_value()) { DCVC_LAUNCH_MMA(true, false); }
            else { DCVC_LAUNCH_MMA(false, false); }
        }
#undef DCVC_LAUNCH_MMA
        return out;
    }

    if (bias.has_value()) {
        const auto bias_contiguous = bias.value().contiguous();
        if (is_tiled_1x1) {
            constexpr int TILE_N = 64;
            const bool use_wide_tile = out_h * out_w >= 128 && out.size(1) >= 1024;
            if (use_wide_tile) {
                constexpr int TILE_M = 64;
                const dim3 grid_dim(ceil_div_int64(out_h * out_w, TILE_N),
                                    ceil_div_int64(out.size(1), TILE_M), out.size(0));
                conv2d_int16_1x1_tiled_kernel<true, TILE_M, 8><<<grid_dim, THREADS, 0, stream>>>(
                    out.data_ptr<int16_t>(), x_contiguous.data_ptr<int16_t>(),
                    weight_contiguous.data_ptr<int16_t>(), bias_contiguous.data_ptr<int16_t>(),
                    out.size(1), out_h * out_w, x_contiguous.size(1));
            } else {
                constexpr int TILE_M = 32;
                const dim3 grid_dim(ceil_div_int64(out_h * out_w, TILE_N),
                                    ceil_div_int64(out.size(1), TILE_M), out.size(0));
                conv2d_int16_1x1_tiled_kernel<true, TILE_M, 4><<<grid_dim, THREADS, 0, stream>>>(
                    out.data_ptr<int16_t>(), x_contiguous.data_ptr<int16_t>(),
                    weight_contiguous.data_ptr<int16_t>(), bias_contiguous.data_ptr<int16_t>(),
                    out.size(1), out_h * out_w, x_contiguous.size(1));
            }
        } else if (is_tiled_3x3) {
            constexpr int TILE_M = 32;
            constexpr int TILE_N = 32;
            constexpr int TILED_THREADS = 128;
            const dim3 grid_dim(ceil_div_int64(out_h * out_w, TILE_N),
                                ceil_div_int64(out.size(1), TILE_M), out.size(0));
            conv2d_int16_3x3_tiled_kernel<true><<<grid_dim, TILED_THREADS, 0, stream>>>(
                out.data_ptr<int16_t>(), x_contiguous.data_ptr<int16_t>(), weight_contiguous.data_ptr<int16_t>(),
                bias_contiguous.data_ptr<int16_t>(), out.size(1), out_h * out_w, x_contiguous.size(1),
                x_contiguous.size(2), x_contiguous.size(3), out_w);
        } else if (is_1x1) {
            conv2d_int16_1x1_kernel<true><<<ceil_div_int64(out_numel, THREADS), THREADS, 0, stream>>>(
                out.data_ptr<int16_t>(), x_contiguous.data_ptr<int16_t>(), weight_contiguous.data_ptr<int16_t>(),
                bias_contiguous.data_ptr<int16_t>(), out_numel, out.size(1), out.size(2), out.size(3),
                x_contiguous.size(1), x_contiguous.size(2), x_contiguous.size(3),
                stride_h, stride_w, pad_h, pad_w, groups);
        } else if (is_depthwise_3x3) {
            conv2d_int16_depthwise_3x3_kernel<true><<<ceil_div_int64(out_numel, THREADS), THREADS, 0, stream>>>(
                out.data_ptr<int16_t>(), x_contiguous.data_ptr<int16_t>(), weight_contiguous.data_ptr<int16_t>(),
                bias_contiguous.data_ptr<int16_t>(), out_numel, out.size(1), out.size(2), out.size(3),
                x_contiguous.size(2), x_contiguous.size(3));
        } else if (is_stride2_2x2) {
            conv2d_int16_stride2_2x2_kernel<true><<<ceil_div_int64(out_numel, THREADS), THREADS, 0, stream>>>(
                out.data_ptr<int16_t>(), x_contiguous.data_ptr<int16_t>(), weight_contiguous.data_ptr<int16_t>(),
                bias_contiguous.data_ptr<int16_t>(), out_numel, out.size(1), out.size(2), out.size(3),
                x_contiguous.size(1), x_contiguous.size(2), x_contiguous.size(3));
        } else if (is_stride2_3x3) {
            conv2d_int16_stride2_3x3_kernel<true><<<ceil_div_int64(out_numel, THREADS), THREADS, 0, stream>>>(
                out.data_ptr<int16_t>(), x_contiguous.data_ptr<int16_t>(), weight_contiguous.data_ptr<int16_t>(),
                bias_contiguous.data_ptr<int16_t>(), out_numel, out.size(1), out.size(2), out.size(3),
                x_contiguous.size(1), x_contiguous.size(2), x_contiguous.size(3));
        } else {
            const dim3 block_dim(16, 16);
            const dim3 grid_dim((out_w + block_dim.x - 1) / block_dim.x,
                                (out_h + block_dim.y - 1) / block_dim.y,
                                out.size(0) * out.size(1));
            conv2d_int16_kernel<true><<<grid_dim, block_dim, 0, stream>>>(
                out.packed_accessor32<int16_t, 4, torch::RestrictPtrTraits>(),
                x_contiguous.packed_accessor32<int16_t, 4, torch::RestrictPtrTraits>(),
                weight_contiguous.packed_accessor32<int16_t, 4, torch::RestrictPtrTraits>(),
                bias_contiguous.packed_accessor32<int16_t, 1, torch::RestrictPtrTraits>(),
                stride_h, stride_w, pad_h, pad_w, groups);
        }
    } else {
        auto empty_bias = torch::empty({ weight_contiguous.size(0) }, x_contiguous.options());
        if (is_tiled_1x1) {
            constexpr int TILE_N = 64;
            const bool use_wide_tile = out_h * out_w >= 128 && out.size(1) >= 1024;
            if (use_wide_tile) {
                constexpr int TILE_M = 64;
                const dim3 grid_dim(ceil_div_int64(out_h * out_w, TILE_N),
                                    ceil_div_int64(out.size(1), TILE_M), out.size(0));
                conv2d_int16_1x1_tiled_kernel<false, TILE_M, 8><<<grid_dim, THREADS, 0, stream>>>(
                    out.data_ptr<int16_t>(), x_contiguous.data_ptr<int16_t>(),
                    weight_contiguous.data_ptr<int16_t>(), empty_bias.data_ptr<int16_t>(),
                    out.size(1), out_h * out_w, x_contiguous.size(1));
            } else {
                constexpr int TILE_M = 32;
                const dim3 grid_dim(ceil_div_int64(out_h * out_w, TILE_N),
                                    ceil_div_int64(out.size(1), TILE_M), out.size(0));
                conv2d_int16_1x1_tiled_kernel<false, TILE_M, 4><<<grid_dim, THREADS, 0, stream>>>(
                    out.data_ptr<int16_t>(), x_contiguous.data_ptr<int16_t>(),
                    weight_contiguous.data_ptr<int16_t>(), empty_bias.data_ptr<int16_t>(),
                    out.size(1), out_h * out_w, x_contiguous.size(1));
            }
        } else if (is_tiled_3x3) {
            constexpr int TILE_M = 32;
            constexpr int TILE_N = 32;
            constexpr int TILED_THREADS = 128;
            const dim3 grid_dim(ceil_div_int64(out_h * out_w, TILE_N),
                                ceil_div_int64(out.size(1), TILE_M), out.size(0));
            conv2d_int16_3x3_tiled_kernel<false><<<grid_dim, TILED_THREADS, 0, stream>>>(
                out.data_ptr<int16_t>(), x_contiguous.data_ptr<int16_t>(), weight_contiguous.data_ptr<int16_t>(),
                empty_bias.data_ptr<int16_t>(), out.size(1), out_h * out_w, x_contiguous.size(1),
                x_contiguous.size(2), x_contiguous.size(3), out_w);
        } else if (is_1x1) {
            conv2d_int16_1x1_kernel<false><<<ceil_div_int64(out_numel, THREADS), THREADS, 0, stream>>>(
                out.data_ptr<int16_t>(), x_contiguous.data_ptr<int16_t>(), weight_contiguous.data_ptr<int16_t>(),
                empty_bias.data_ptr<int16_t>(), out_numel, out.size(1), out.size(2), out.size(3),
                x_contiguous.size(1), x_contiguous.size(2), x_contiguous.size(3),
                stride_h, stride_w, pad_h, pad_w, groups);
        } else if (is_depthwise_3x3) {
            conv2d_int16_depthwise_3x3_kernel<false><<<ceil_div_int64(out_numel, THREADS), THREADS, 0, stream>>>(
                out.data_ptr<int16_t>(), x_contiguous.data_ptr<int16_t>(), weight_contiguous.data_ptr<int16_t>(),
                empty_bias.data_ptr<int16_t>(), out_numel, out.size(1), out.size(2), out.size(3),
                x_contiguous.size(2), x_contiguous.size(3));
        } else if (is_stride2_2x2) {
            conv2d_int16_stride2_2x2_kernel<false><<<ceil_div_int64(out_numel, THREADS), THREADS, 0, stream>>>(
                out.data_ptr<int16_t>(), x_contiguous.data_ptr<int16_t>(), weight_contiguous.data_ptr<int16_t>(),
                empty_bias.data_ptr<int16_t>(), out_numel, out.size(1), out.size(2), out.size(3),
                x_contiguous.size(1), x_contiguous.size(2), x_contiguous.size(3));
        } else if (is_stride2_3x3) {
            conv2d_int16_stride2_3x3_kernel<false><<<ceil_div_int64(out_numel, THREADS), THREADS, 0, stream>>>(
                out.data_ptr<int16_t>(), x_contiguous.data_ptr<int16_t>(), weight_contiguous.data_ptr<int16_t>(),
                empty_bias.data_ptr<int16_t>(), out_numel, out.size(1), out.size(2), out.size(3),
                x_contiguous.size(1), x_contiguous.size(2), x_contiguous.size(3));
        } else {
            const dim3 block_dim(16, 16);
            const dim3 grid_dim((out_w + block_dim.x - 1) / block_dim.x,
                                (out_h + block_dim.y - 1) / block_dim.y,
                                out.size(0) * out.size(1));
            conv2d_int16_kernel<false><<<grid_dim, block_dim, 0, stream>>>(
                out.packed_accessor32<int16_t, 4, torch::RestrictPtrTraits>(),
                x_contiguous.packed_accessor32<int16_t, 4, torch::RestrictPtrTraits>(),
                weight_contiguous.packed_accessor32<int16_t, 4, torch::RestrictPtrTraits>(),
                empty_bias.packed_accessor32<int16_t, 1, torch::RestrictPtrTraits>(),
                stride_h, stride_w, pad_h, pad_w, groups);
        }
    }
    return out;
}

torch::Tensor add_bias_int16_cuda(const torch::Tensor& x, const torch::Tensor& bias)
{
    check_cuda_int16(x, "x");
    check_cuda_int16(bias, "bias");
    check_4d(x, "x");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    TORCH_CHECK(bias.size(0) == x.size(1), "bias size must equal the channel dimension");

    const auto x_contiguous = x.contiguous();
    const auto bias_contiguous = bias.contiguous();
    auto out = torch::empty_like(x_contiguous);
    const int64_t N = x_contiguous.numel();
    const int HW = x_contiguous.size(2) * x_contiguous.size(3);
    auto stream = c10::cuda::getCurrentCUDAStream();
    add_bias_int16_kernel<<<ceil_div_int64(N, THREADS), THREADS, 0, stream>>>(
        out.data_ptr<int16_t>(), x_contiguous.data_ptr<int16_t>(), bias_contiguous.data_ptr<int16_t>(),
        N, x_contiguous.size(1), HW);
    return out;
}

torch::Tensor add_tensors_int16_cuda(const torch::Tensor& x, const torch::Tensor& y)
{
    check_cuda_int16(x, "x");
    check_cuda_int16(y, "y");
    check_same_shape(x, y, "x", "y");

    const auto x_contiguous = x.contiguous();
    const auto y_contiguous = y.contiguous();
    auto out = torch::empty_like(x_contiguous);
    const int64_t N = x_contiguous.numel();
    auto stream = c10::cuda::getCurrentCUDAStream();
    add_tensors_int16_kernel<<<ceil_div_int64(N, THREADS), THREADS, 0, stream>>>(
        out.data_ptr<int16_t>(), x_contiguous.data_ptr<int16_t>(), y_contiguous.data_ptr<int16_t>(), N);
    return out;
}

torch::Tensor conv2d_int16_residual_cuda(const torch::Tensor& x, const torch::Tensor& weight,
    const torch::optional<torch::Tensor>& bias, const torch::Tensor& residual, const int tile_n)
{
    check_cuda_int16(x, "x");
    check_cuda_int16(weight, "weight");
    check_cuda_int16(residual, "residual");
    check_4d(x, "x");
    check_4d(weight, "weight");
    check_4d(residual, "residual");
    TORCH_CHECK(weight.device() == x.device() && residual.device() == x.device(),
                "all inputs must be on the same CUDA device");
    TORCH_CHECK(weight.size(2) == 1 && weight.size(3) == 1 && weight.size(1) == x.size(1),
                "expected a dense 1x1 convolution");
    TORCH_CHECK(residual.size(0) == x.size(0) && residual.size(1) == weight.size(0) &&
                residual.size(2) == x.size(2) && residual.size(3) == x.size(3),
                "residual must match the convolution output shape");
    TORCH_CHECK(tile_n == 16 || tile_n == 32 || tile_n == 64, "tile_n must be 16, 32 or 64");
    if (bias.has_value()) {
        check_cuda_int16(bias.value(), "bias");
        TORCH_CHECK(bias.value().device() == x.device() && bias.value().dim() == 1 &&
                    bias.value().size(0) == weight.size(0), "invalid convolution bias");
    }
    const c10::cuda::CUDAGuard guard(x.device());
    if (!int16_mma_supported(x.get_device()))
        return add_tensors_int16_cuda(conv2d_int16_cuda(x, weight, bias, 1, 1, 0, 0, 1), residual);
    const auto xc = x.contiguous(), wc = weight.contiguous(), rc = residual.contiguous();
    const auto bc = bias.has_value() ? bias.value().contiguous() : torch::Tensor();
    const auto bp = bias.has_value() ? bc.data_ptr<int16_t>() : nullptr;
    auto out = torch::empty_like(rc);
    if (out.numel() == 0) return out;
    const dim3 grid(ceil_div_int64(x.size(2)*x.size(3), tile_n),
                    ceil_div_int64(weight.size(0), 64), x.size(0));
    auto stream = c10::cuda::getCurrentCUDAStream();
#define DCVC_RESIDUAL_MMA(BIAS, BN) \
    conv2d_int16_mma_kernel<BIAS, false, BN, true><<<grid, 128, 0, stream>>>( \
        out.data_ptr<int16_t>(), xc.data_ptr<int16_t>(), wc.data_ptr<int16_t>(), bp, \
        weight.size(0), x.size(2)*x.size(3), x.size(1), x.size(2), x.size(3), rc.data_ptr<int16_t>())
#define DCVC_RESIDUAL_TILE(BN) \
    if (bias.has_value()) { DCVC_RESIDUAL_MMA(true, BN); } \
    else { DCVC_RESIDUAL_MMA(false, BN); }
    if (tile_n == 16) { DCVC_RESIDUAL_TILE(16); }
    else if (tile_n == 64) { DCVC_RESIDUAL_TILE(64); }
    else { DCVC_RESIDUAL_TILE(32); }
#undef DCVC_RESIDUAL_TILE
#undef DCVC_RESIDUAL_MMA
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor mul_feature_scale_int16_cuda(const torch::Tensor& x, const torch::Tensor& scale)
{
    check_cuda_int16(x, "x");
    check_cuda_int16(scale, "scale");
    check_broadcastable_4d(x, scale, "x", "scale");

    const auto x_contiguous = x.contiguous();
    const auto scale_contiguous = scale.contiguous();
    auto out = torch::empty_like(x_contiguous);
    const int64_t N = x_contiguous.numel();
    auto stream = c10::cuda::getCurrentCUDAStream();
    mul_feature_scale_int16_kernel<<<ceil_div_int64(N, THREADS), THREADS, 0, stream>>>(
        out.data_ptr<int16_t>(), x_contiguous.data_ptr<int16_t>(), scale_contiguous.data_ptr<int16_t>(),
        N, x_contiguous.size(0), x_contiguous.size(1), x_contiguous.size(2), x_contiguous.size(3),
        scale_contiguous.size(0), scale_contiguous.size(1), scale_contiguous.size(2), scale_contiguous.size(3));
    return out;
}

std::tuple<torch::Tensor, torch::Tensor> reciprocal_scale_int16_cuda(const torch::Tensor& q_dec,
                                                                     const float min_val)
{
    check_cuda_int16(q_dec, "q_dec");
    const auto q_dec_contiguous = q_dec.contiguous();
    auto out_q_dec = torch::empty_like(q_dec_contiguous);
    auto out_recip = torch::empty_like(q_dec_contiguous);
    const int64_t N = q_dec_contiguous.numel();
    const int32_t min_q = static_cast<int32_t>(roundf(min_val * FEATURE_SCALE));
    auto stream = c10::cuda::getCurrentCUDAStream();
    reciprocal_scale_int16_kernel<<<ceil_div_int64(N, THREADS), THREADS, 0, stream>>>(
        out_q_dec.data_ptr<int16_t>(), out_recip.data_ptr<int16_t>(),
        q_dec_contiguous.data_ptr<int16_t>(), N, min_q);
    return std::make_tuple(out_q_dec, out_recip);
}

torch::Tensor apply_lut_int16_cuda(const torch::Tensor& x, const torch::Tensor& lut)
{
    check_cuda_int16(x, "x");
    check_cuda_int16(lut, "lut");
    TORCH_CHECK(lut.dim() == 1, "lut must be 1D");
    TORCH_CHECK(lut.size(0) == 65536, "lut must have 65536 entries");

    const auto x_contiguous = x.contiguous();
    const auto lut_contiguous = lut.contiguous();
    auto out = torch::empty_like(x_contiguous);
    const int64_t N = x_contiguous.numel();
    auto stream = c10::cuda::getCurrentCUDAStream();
    apply_lut_int16_kernel<<<ceil_div_int64(N, THREADS), THREADS, 0, stream>>>(
        out.data_ptr<int16_t>(), x_contiguous.data_ptr<int16_t>(), lut_contiguous.data_ptr<int16_t>(), N);
    return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
process_with_mask_int16_cuda(const torch::Tensor& y, const torch::Tensor& scales, const torch::Tensor& means,
                             const torch::Tensor& mask, const int32_t force_zero_thres)
{
    check_cuda_int16(y, "y");
    check_cuda_int16(scales, "scales");
    check_cuda_int16(means, "means");
    check_cuda_int16(mask, "mask");
    check_same_shape(y, scales, "y", "scales");
    check_same_shape(y, means, "y", "means");
    check_same_shape(y, mask, "y", "mask");

    const auto y_contiguous = y.contiguous();
    const auto scales_contiguous = scales.contiguous();
    const auto means_contiguous = means.contiguous();
    const auto mask_contiguous = mask.contiguous();
    auto y_res = torch::empty_like(y_contiguous);
    auto y_q = torch::empty_like(y_contiguous);
    auto y_hat = torch::empty_like(y_contiguous);
    auto s_hat = torch::empty_like(y_contiguous);
    const int64_t N = y_contiguous.numel();
    auto stream = c10::cuda::getCurrentCUDAStream();
    process_with_mask_int16_kernel<<<ceil_div_int64(N, THREADS), THREADS, 0, stream>>>(
        y_res.data_ptr<int16_t>(), y_q.data_ptr<int16_t>(), y_hat.data_ptr<int16_t>(),
        s_hat.data_ptr<int16_t>(), y_contiguous.data_ptr<int16_t>(), scales_contiguous.data_ptr<int16_t>(),
        means_contiguous.data_ptr<int16_t>(), mask_contiguous.data_ptr<int16_t>(), N, force_zero_thres);
    return std::make_tuple(y_res, y_q, y_hat, s_hat);
}

void combine_for_reading_int16_cuda(torch::Tensor& out, const torch::Tensor& x, const torch::Tensor& mask,
                                    const int parts)
{
    check_cuda_int16(out, "out");
    check_cuda_int16(x, "x");
    check_cuda_int16(mask, "mask");
    check_4d(out, "out");
    check_4d(x, "x");
    check_4d(mask, "mask");
    check_same_shape(x, mask, "x", "mask");
    TORCH_CHECK(parts > 0, "parts must be positive");
    TORCH_CHECK(x.size(1) % parts == 0, "channel dimension must be divisible by parts");
    TORCH_CHECK(out.size(0) == x.size(0) && out.size(1) == x.size(1) / parts &&
                out.size(2) == x.size(2) && out.size(3) == x.size(3),
                "out shape is incompatible with x and parts");

    auto stream = c10::cuda::getCurrentCUDAStream();
    combine_for_reading_int16_kernel<<<ceil_div_int64(out.numel(), THREADS), THREADS, 0, stream>>>(
        out.data_ptr<int16_t>(), x.contiguous().data_ptr<int16_t>(), mask.contiguous().data_ptr<int16_t>(),
        out.numel(), out.size(1), out.size(2), out.size(3), parts);
}

void restore_y_parts_int16_cuda(torch::Tensor& out, const torch::Tensor& y, const torch::Tensor& means,
                                const torch::Tensor& mask, const int parts)
{
    check_cuda_int16(out, "out");
    check_cuda_int16(y, "y");
    check_cuda_int16(means, "means");
    check_cuda_int16(mask, "mask");
    check_4d(out, "out");
    check_4d(y, "y");
    check_4d(means, "means");
    check_4d(mask, "mask");
    check_same_shape(out, means, "out", "means");
    check_same_shape(out, mask, "out", "mask");
    TORCH_CHECK(parts > 0, "parts must be positive");
    TORCH_CHECK(out.size(1) == y.size(1) * parts, "out channels must equal y channels times parts");
    TORCH_CHECK(out.size(0) == y.size(0) && out.size(2) == y.size(2) && out.size(3) == y.size(3),
                "out spatial shape must match y");

    auto stream = c10::cuda::getCurrentCUDAStream();
    restore_y_parts_int16_kernel<<<ceil_div_int64(out.numel(), THREADS), THREADS, 0, stream>>>(
        out.data_ptr<int16_t>(), y.contiguous().data_ptr<int16_t>(), means.contiguous().data_ptr<int16_t>(),
        mask.contiguous().data_ptr<int16_t>(), out.numel(), out.size(1), y.size(1), out.size(2), out.size(3));
}

std::tuple<torch::Tensor, torch::Tensor> build_index_dec_int16_cuda(const torch::Tensor& scales,
                                                                    const torch::Tensor& lut,
                                                                    const int32_t skip_thres)
{
    check_cuda_int16(scales, "scales");
    TORCH_CHECK(lut.is_cuda(), "lut must be a CUDA tensor");
    TORCH_CHECK(lut.scalar_type() == torch::kUInt8, "lut must be uint8");
    TORCH_CHECK(lut.dim() == 1 && lut.size(0) == 65536, "lut must have 65536 entries");

    const auto scales_contiguous = scales.contiguous();
    const auto lut_contiguous = lut.contiguous();
    auto out = torch::empty_like(scales_contiguous, scales_contiguous.options().dtype(torch::kUInt8));
    auto stream = c10::cuda::getCurrentCUDAStream();
    if (skip_thres >= 0) {
        auto skip_cond = torch::empty_like(scales_contiguous, scales_contiguous.options().dtype(torch::kBool));
        build_index_dec_int16_kernel<<<ceil_div_int64(scales_contiguous.numel(), THREADS), THREADS, 0, stream>>>(
            out.data_ptr<uint8_t>(), skip_cond.data_ptr<bool>(), scales_contiguous.data_ptr<int16_t>(),
            lut_contiguous.data_ptr<uint8_t>(), scales_contiguous.numel(), skip_thres, true);
        return std::make_tuple(out, skip_cond);
    }

    auto empty_skip = torch::empty({ 0 }, scales_contiguous.options().dtype(torch::kBool));
    build_index_dec_int16_kernel<<<ceil_div_int64(scales_contiguous.numel(), THREADS), THREADS, 0, stream>>>(
        out.data_ptr<uint8_t>(), nullptr, scales_contiguous.data_ptr<int16_t>(), lut_contiguous.data_ptr<uint8_t>(),
        scales_contiguous.numel(), -1, false);
    return std::make_tuple(out, empty_skip);
}

torch::Tensor build_index_enc_int16_cuda(const torch::Tensor& symbols, const torch::Tensor& scales,
                                         const torch::Tensor& lut, const int32_t skip_thres)
{
    check_cuda_int16(symbols, "symbols");
    check_cuda_int16(scales, "scales");
    check_same_shape(symbols, scales, "symbols", "scales");
    TORCH_CHECK(lut.is_cuda(), "lut must be a CUDA tensor");
    TORCH_CHECK(lut.scalar_type() == torch::kUInt8, "lut must be uint8");
    TORCH_CHECK(lut.dim() == 1 && lut.size(0) == 65536, "lut must have 65536 entries");

    const auto symbols_contiguous = symbols.contiguous();
    const auto scales_contiguous = scales.contiguous();
    const auto lut_contiguous = lut.contiguous();
    auto out = torch::empty_like(scales_contiguous);
    auto stream = c10::cuda::getCurrentCUDAStream();
    if (skip_thres >= 0) {
        auto skip_cond = torch::empty_like(scales_contiguous, scales_contiguous.options().dtype(torch::kBool));
        build_index_enc_int16_kernel<<<ceil_div_int64(scales_contiguous.numel(), THREADS), THREADS, 0, stream>>>(
            out.data_ptr<int16_t>(), skip_cond.data_ptr<bool>(), symbols_contiguous.data_ptr<int16_t>(),
            scales_contiguous.data_ptr<int16_t>(), lut_contiguous.data_ptr<uint8_t>(),
            scales_contiguous.numel(), skip_thres, true);
        return torch::masked_select(out, skip_cond);
    }

    build_index_enc_int16_kernel<<<ceil_div_int64(scales_contiguous.numel(), THREADS), THREADS, 0, stream>>>(
        out.data_ptr<int16_t>(), nullptr, symbols_contiguous.data_ptr<int16_t>(),
        scales_contiguous.data_ptr<int16_t>(), lut_contiguous.data_ptr<uint8_t>(),
        scales_contiguous.numel(), -1, false);
    return out;
}

torch::Tensor add_and_multiply_int16_cuda(const torch::Tensor& x0, const torch::Tensor& x1,
                                          const torch::Tensor& q)
{
    check_cuda_int16(x0, "x0");
    check_cuda_int16(x1, "x1");
    check_cuda_int16(q, "q");
    check_same_shape(x0, x1, "x0", "x1");
    check_broadcastable_4d(x0, q, "x0", "q");

    const auto x0_contiguous = x0.contiguous();
    const auto x1_contiguous = x1.contiguous();
    const auto q_contiguous = q.contiguous();
    auto out = torch::empty_like(x0_contiguous);
    const int64_t N = x0_contiguous.numel();
    auto stream = c10::cuda::getCurrentCUDAStream();
    add_and_multiply_int16_kernel<<<ceil_div_int64(N, THREADS), THREADS, 0, stream>>>(
        out.data_ptr<int16_t>(), x0_contiguous.data_ptr<int16_t>(), x1_contiguous.data_ptr<int16_t>(),
        q_contiguous.data_ptr<int16_t>(), N,
        x0_contiguous.size(0), x0_contiguous.size(1), x0_contiguous.size(2), x0_contiguous.size(3),
        q_contiguous.size(0), q_contiguous.size(1), q_contiguous.size(2), q_contiguous.size(3));
    return out;
}

torch::Tensor bias_quant_int16_cuda(const torch::Tensor& x, const torch::Tensor& bias,
                                    const torch::Tensor& quant_step)
{
    check_cuda_int16(x, "x");
    check_cuda_int16(bias, "bias");
    check_cuda_int16(quant_step, "quant_step");
    check_4d(x, "x");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    TORCH_CHECK(bias.size(0) == x.size(1), "bias size must equal the channel dimension");
    check_broadcastable_4d(x, quant_step, "x", "quant_step");

    const auto x_contiguous = x.contiguous();
    const auto bias_contiguous = bias.contiguous();
    const auto quant_step_contiguous = quant_step.contiguous();
    auto out = torch::empty_like(x_contiguous);
    const int64_t N = x_contiguous.numel();
    const int HW = x_contiguous.size(2) * x_contiguous.size(3);
    auto stream = c10::cuda::getCurrentCUDAStream();
    bias_quant_int16_kernel<<<ceil_div_int64(N, THREADS), THREADS, 0, stream>>>(
        out.data_ptr<int16_t>(), x_contiguous.data_ptr<int16_t>(), bias_contiguous.data_ptr<int16_t>(),
        quant_step_contiguous.data_ptr<int16_t>(), N, x_contiguous.size(1), HW,
        x_contiguous.size(0), x_contiguous.size(2), x_contiguous.size(3),
        quant_step_contiguous.size(0), quant_step_contiguous.size(1),
        quant_step_contiguous.size(2), quant_step_contiguous.size(3));
    return out;
}

torch::Tensor wsilu_chunk_add_int16_cuda(const torch::Tensor& x, const torch::Tensor& lut)
{
    check_cuda_int16(x, "x");
    check_cuda_int16(lut, "lut");
    check_4d(x, "x");
    TORCH_CHECK(x.size(1) % 2 == 0, "channel dimension must be divisible by 2");
    TORCH_CHECK(lut.dim() == 1, "lut must be 1D");
    TORCH_CHECK(lut.size(0) == 65536, "lut must have 65536 entries");

    const auto x_contiguous = x.contiguous();
    const auto lut_contiguous = lut.contiguous();
    auto out = torch::empty({ x_contiguous.size(0), x_contiguous.size(1) / 2,
                              x_contiguous.size(2), x_contiguous.size(3) },
                            x_contiguous.options());
    auto stream = c10::cuda::getCurrentCUDAStream();
    wsilu_chunk_add_int16_kernel<<<ceil_div_int64(out.numel(), THREADS), THREADS, 0, stream>>>(
        out.data_ptr<int16_t>(), x_contiguous.data_ptr<int16_t>(), lut_contiguous.data_ptr<int16_t>(),
        out.numel(), out.size(1), out.size(2), out.size(3));
    return out;
}

torch::Tensor bias_wsilu_depthwise_conv2d_int16_cuda(const torch::Tensor& x, const torch::Tensor& weight,
                                                     const torch::Tensor& bias, const torch::Tensor& lut)
{
    check_cuda_int16(x, "x");
    check_cuda_int16(weight, "weight");
    check_cuda_int16(bias, "bias");
    check_cuda_int16(lut, "lut");
    check_4d(x, "x");
    check_4d(weight, "weight");
    TORCH_CHECK(weight.size(0) == x.size(1) && weight.size(1) == 1 &&
                weight.size(2) == 3 && weight.size(3) == 3,
                "weight must be depthwise 3x3 with channel dimension matching x");
    TORCH_CHECK(bias.dim() == 1 && bias.size(0) == x.size(1), "bias size must equal x channels");
    TORCH_CHECK(lut.dim() == 1 && lut.size(0) == 65536, "lut must have 65536 entries");

    const auto x_contiguous = x.contiguous();
    const auto weight_contiguous = weight.contiguous();
    const auto bias_contiguous = bias.contiguous();
    const auto lut_contiguous = lut.contiguous();
    auto out = torch::empty_like(x_contiguous);
    auto stream = c10::cuda::getCurrentCUDAStream();
    bias_wsilu_depthwise_conv2d_int16_kernel<<<ceil_div_int64(out.numel(), THREADS), THREADS, 0, stream>>>(
        out.data_ptr<int16_t>(), x_contiguous.data_ptr<int16_t>(), weight_contiguous.data_ptr<int16_t>(),
        bias_contiguous.data_ptr<int16_t>(), lut_contiguous.data_ptr<int16_t>(),
        out.numel(), out.size(1), out.size(2), out.size(3));
    return out;
}

torch::Tensor bias_pixel_shuffle_2_int16_cuda(const torch::Tensor& x, const torch::Tensor& bias)
{
    check_cuda_int16(x, "x");
    check_cuda_int16(bias, "bias");
    check_4d(x, "x");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    TORCH_CHECK(x.size(1) % 4 == 0, "channel dimension must be divisible by 4");
    TORCH_CHECK(bias.size(0) == x.size(1), "bias size must equal the channel dimension");

    const auto x_contiguous = x.contiguous();
    const auto bias_contiguous = bias.contiguous();
    auto out = torch::empty({ x_contiguous.size(0), x_contiguous.size(1) / 4,
                              x_contiguous.size(2) * 2, x_contiguous.size(3) * 2 },
                            x_contiguous.options());
    auto stream = c10::cuda::getCurrentCUDAStream();
    bias_pixel_shuffle_2_int16_kernel<<<ceil_div_int64(out.numel(), THREADS), THREADS, 0, stream>>>(
        out.data_ptr<int16_t>(), x_contiguous.data_ptr<int16_t>(), bias_contiguous.data_ptr<int16_t>(),
        out.numel(), out.size(0), out.size(1), out.size(2), out.size(3),
        x_contiguous.size(2), x_contiguous.size(3));
    return out;
}

torch::Tensor bias_pixel_shuffle_8_int16_cuda(const torch::Tensor& x, const torch::Tensor& bias)
{
    check_cuda_int16(x, "x");
    check_cuda_int16(bias, "bias");
    check_4d(x, "x");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    TORCH_CHECK(x.size(1) % 64 == 0, "channel dimension must be divisible by 64");
    TORCH_CHECK(bias.size(0) == x.size(1), "bias size must equal the channel dimension");

    const auto x_contiguous = x.contiguous();
    const auto bias_contiguous = bias.contiguous();
    auto out = torch::empty({ x_contiguous.size(0), x_contiguous.size(1) / 64,
                              x_contiguous.size(2) * 8, x_contiguous.size(3) * 8 },
                            x_contiguous.options());
    auto stream = c10::cuda::getCurrentCUDAStream();
    bias_pixel_shuffle_8_int16_kernel<<<ceil_div_int64(out.numel(), THREADS), THREADS, 0, stream>>>(
        out.data_ptr<int16_t>(), x_contiguous.data_ptr<int16_t>(), bias_contiguous.data_ptr<int16_t>(),
        out.numel(), out.size(0), out.size(1), out.size(2), out.size(3),
        x_contiguous.size(2), x_contiguous.size(3));
    return out;
}

namespace {
__global__ void round_and_to_int8_int16_kernel(int16_t* feature, int8_t* symbols,
                                               const int16_t* x, int64_t N)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int32_t q = max(-128, min(127, round_divide_int(static_cast<int32_t>(x[i]), FEATURE_SCALE)));
    symbols[i] = static_cast<int8_t>(q);
    feature[i] = clip_to_int16(q*FEATURE_SCALE);
}
}  // namespace

std::tuple<torch::Tensor, torch::Tensor> round_and_to_int8_int16_cuda(const torch::Tensor& x)
{
    check_cuda_int16(x, "x");
    const auto input = x.contiguous();
    auto feature = torch::empty_like(input);
    auto symbols = torch::empty(input.sizes(), input.options().dtype(torch::kInt8));
    if (input.numel() > 0) {
        round_and_to_int8_int16_kernel<<<ceil_div_int64(input.numel(), THREADS), THREADS, 0,
                                       c10::cuda::getCurrentCUDAStream()>>>(
            feature.data_ptr<int16_t>(), symbols.data_ptr<int8_t>(), input.data_ptr<int16_t>(), input.numel());
    }
    return {feature, symbols};
}
