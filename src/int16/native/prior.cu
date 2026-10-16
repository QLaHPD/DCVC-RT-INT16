// Exact UF spatial-partition gather/quantize/restore without int64 temporaries.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {
constexpr int SKIPPED = 65536;  // Outside the packed int16 symbol/index range.

__device__ int16_t saturate(int value) {
    return value < -32768 ? -32768 : (value > 32767 ? 32767 : value);
}

__global__ void encode_partition(int32_t* packed, int16_t* restored,
    const int16_t* y, const int16_t* scales, const int16_t* means,
    const int64_t* positions, const uint8_t* table, int count, int threshold) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int64_t p = positions[i];
    int symbol = 0;
    bool active = threshold == 0 || scales[p] > threshold;
    if (active) {
        int residual = static_cast<int>(y[p]) - means[p];
        symbol = residual < 0 ? -((-residual + 256) / 512) : (residual + 256) / 512;
        symbol = symbol < -128 ? -128 : (symbol > 127 ? 127 : symbol);
    }
    packed[i] = active ? symbol * 256 + table[static_cast<int>(scales[p]) + 32768] : SKIPPED;
    restored[p] = saturate(symbol * 512 + means[p]);
}

__global__ void partition_indexes(int16_t* indexes, const int16_t* scales,
    const int64_t* positions, const uint8_t* table, int count, int threshold) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int scale = scales[positions[i]];
    indexes[i] = threshold == 0 || scale > threshold ? table[scale + 32768] : -1;
}

__global__ void restore_partition(int16_t* restored, const int8_t* symbols,
    const int16_t* means, const int64_t* positions, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int64_t p = positions[i];
    restored[p] = saturate(static_cast<int>(symbols[i]) * 512 + means[p]);
}

void check_values(const torch::Tensor& values, const torch::Tensor& positions) {
    TORCH_CHECK(values.is_cuda() && values.scalar_type() == at::kShort,
                "prior values must be CUDA int16");
    TORCH_CHECK(values.numel() > 0 && values.numel() <= INT32_MAX, "unsupported prior size");
    TORCH_CHECK(positions.device() == values.device() && positions.scalar_type() == at::kLong &&
                positions.dim() == 1 && positions.is_contiguous() && positions.numel() > 0 &&
                positions.numel() <= values.numel(), "invalid prior positions");
}

void check_table(const torch::Tensor& table, const torch::Tensor& values, int threshold) {
    TORCH_CHECK(table.device() == values.device() && table.scalar_type() == at::kByte &&
                table.dim() == 1 && table.numel() == 65536 && table.is_contiguous(),
                "prior scale table must contain 65536 CUDA bytes");
    TORCH_CHECK(threshold >= 0 && threshold <= 32767, "invalid prior skip threshold");
}

void check_restored(const torch::Tensor& restored, const torch::Tensor& values) {
    TORCH_CHECK(restored.device() == values.device() && restored.scalar_type() == at::kShort &&
                restored.sizes() == values.sizes() && restored.is_contiguous(),
                "invalid prior reconstruction buffer");
}
}

torch::Tensor uf_prior_encode(const torch::Tensor& y, const torch::Tensor& scales,
    const torch::Tensor& means, torch::Tensor restored, const torch::Tensor& positions,
    const torch::Tensor& table, int threshold) {
    check_values(scales, positions);
    check_table(table, scales, threshold);
    check_restored(restored, scales);
    TORCH_CHECK(y.device() == scales.device() && means.device() == scales.device() &&
                y.scalar_type() == at::kShort && means.scalar_type() == at::kShort &&
                y.sizes() == scales.sizes() && means.sizes() == scales.sizes(),
                "invalid prior feature/mean tensors");
    c10::cuda::CUDAGuard guard(scales.device());
    auto input = y.contiguous(), s = scales.contiguous(), m = means.contiguous();
    auto packed = torch::empty(positions.sizes(), scales.options().dtype(at::kInt));
    encode_partition<<<(positions.numel()+255)/256, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        packed.data_ptr<int32_t>(), restored.data_ptr<int16_t>(), input.data_ptr<int16_t>(),
        s.data_ptr<int16_t>(), m.data_ptr<int16_t>(), positions.data_ptr<int64_t>(),
        table.data_ptr<uint8_t>(), positions.numel(), threshold);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return packed;
}

torch::Tensor uf_prior_indexes(const torch::Tensor& scales, const torch::Tensor& positions,
    const torch::Tensor& table, int threshold) {
    check_values(scales, positions);
    check_table(table, scales, threshold);
    c10::cuda::CUDAGuard guard(scales.device());
    auto s = scales.contiguous();
    auto indexes = torch::empty(positions.sizes(), scales.options());
    partition_indexes<<<(positions.numel()+255)/256, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        indexes.data_ptr<int16_t>(), s.data_ptr<int16_t>(), positions.data_ptr<int64_t>(),
        table.data_ptr<uint8_t>(), positions.numel(), threshold);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return indexes;
}

void uf_prior_restore(const torch::Tensor& symbols, const torch::Tensor& means,
    torch::Tensor restored, const torch::Tensor& positions) {
    check_values(means, positions);
    check_restored(restored, means);
    TORCH_CHECK(symbols.device() == means.device() && symbols.scalar_type() == at::kChar &&
                symbols.sizes() == positions.sizes() && symbols.is_contiguous(),
                "prior symbols must be contiguous CUDA int8");
    c10::cuda::CUDAGuard guard(means.device());
    auto m = means.contiguous();
    restore_partition<<<(positions.numel()+255)/256, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        restored.data_ptr<int16_t>(), symbols.data_ptr<int8_t>(), m.data_ptr<int16_t>(),
        positions.data_ptr<int64_t>(), positions.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
