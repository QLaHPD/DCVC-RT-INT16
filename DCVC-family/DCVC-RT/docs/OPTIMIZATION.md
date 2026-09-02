# Kernel optimization evidence

The optimization targeted the highest-cost INT16 convolution shapes observed in one I-frame, one reset P-frame, and one steady P-frame on an NVIDIA Jetson Orin.

## Implemented changes

- Shared-memory tiled implicit GEMM for frequent dense `1x1` convolutions.
- Shared-memory tiled implicit GEMM for a frequent dense pad-1 `3x3` convolution.
- Preservation of K traversal, INT32 low-bit wrap, rounding, and INT16 clamp behavior.
- Generic fallback for all shapes outside the optimized dispatch.
- Removal of redundant per-frame device-wide synchronization.
- Lighter CUDA stream header dependencies for more portable extension compilation.

## Local results

| Measurement | Reference | Optimized | Change |
| --- | ---: | ---: | ---: |
| Encoder loop | 21.9 fps | 38.6 fps | +76% |
| Encoder command wall time | ~25.9 s | ~21.2 s | ~18% lower |
| Decoder loop | 21.9 fps | 37.6 fps | +72% |
| Decoder command wall time | ~26.7 s | ~20.9 s | ~22% lower |

The optimized and reference INT16 implementations produced identical encoded bytes and decoded YUV bytes in the local end-to-end fixture. Sixteen representative convolution shapes also matched exactly, including randomized values designed to exercise accumulator wrap.

These measurements describe one small synthetic-equivalent engineering workload and one device. They are not general performance claims and should not be compared directly with the paper's 1080p A100 measurements.

## Reproducible tools

`benchmarks/benchmark_int16_conv.py` generates deterministic tensors and can save or compare exact reference outputs. `benchmarks/profile_int16_conv_shapes.py` and `benchmarks/profile_steady_frame.py` use deterministic synthetic frames; no third-party video is required or distributed.
