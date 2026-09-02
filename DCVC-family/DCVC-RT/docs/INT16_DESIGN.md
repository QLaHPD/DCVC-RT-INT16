# INT16 runtime design

## Representation

The runtime uses two fixed scales:

| Quantity | Storage | Scale | Approximate represented value |
| --- | --- | ---: | --- |
| Network feature or bias | signed INT16 | 512 | `integer / 512` |
| Convolution weight | signed INT16 | 8192 | `integer / 8192` |

Float checkpoints are loaded normally, then convolution weights, biases, and selected model parameters are rounded and clipped into these representations. Prepared CPU tensors are moved to a device once and cached per module/device.

## Arithmetic

Convolutions multiply INT16 inputs and weights and accumulate through INT32 semantics. The feature/weight product is divided by the weight scale using explicit signed round-to-nearest behavior, then clipped to signed INT16. Basic feature arithmetic likewise uses explicit INT32 intermediates before rescaling and clamping.

Nonlinear functions that would otherwise reintroduce device-dependent floating-point behavior are represented by complete INT16-domain lookup tables. Separate tables cover the WSiLU-style activation, entropy-prior sigmoid transform, and scale-to-index conversion.

The extension includes integer operations for:

- dense, grouped, and depthwise convolution
- bias and residual addition
- feature-scale multiplication and reciprocal scale generation
- nonlinear and entropy-index LUT application
- masked quantization and entropy symbol preparation
- fused WSiLU/depthwise/residual patterns
- pixel-shuffle reconstruction paths

## Prepared checkpoint caches

Loading a float checkpoint with INT16 enabled may generate:

```text
<checkpoint-name>.int16prep.pt
```

This cache contains quantized model parameters, prepared entropy state, and lookup tables. It avoids repeating conversion work during later startup. It is derived from the original checkpoint, machine-recreatable, and intentionally ignored by Git.

## CUDA fast paths

`int_kernel.cu` retains general kernels and dispatches two frequent dense shapes to tiled implicit-GEMM implementations:

1. group-1, stride-1, unpadded `1x1`
2. group-1, stride-1, pad-1 `3x3`

Tiles reuse activations and weights from shared memory. The `3x3` path synthesizes boundary zeroes rather than materializing an im2col buffer. Both preserve the reference K traversal and low-32-bit accumulation behavior. Other configurations remain on the reference dispatch.

## Bitstream and decoder compatibility

The fork does not intentionally alter DCVC's NAL/entropy bitstream syntax. Kernel regression tests require exact tensors, and end-to-end testing found the optimized and reference INT16 encoders produced identical bytes.

Predictive neural decoding depends on reconstructed reference features. Therefore, format compatibility alone is weaker than deterministic reconstruction: archival encode/decode should use the INT16 runtime at both endpoints. Compatibility is asserted with the earlier INT16 implementation, not as a guarantee that an arbitrary upstream floating-point build produces identical intermediate features.

## Paper correspondence

Section 4.5 of the CVPR 2025 paper describes training-free 16-bit integerization, feature scale 512, INT32 convolution accumulation, and lookup-table nonlinear functions. This runtime implements those principles around the published checkpoints.

The paper does not prescribe this repository's Python cache format, CUDA dispatch details, extension discovery, fused deployment kernels, FFmpeg pipeline, or archive lifecycle. Those are fork-specific engineering decisions and are not presented as paper results.
