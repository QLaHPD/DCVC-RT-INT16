# Residual fusion and profiling follow-up

## Implementation

Two stable convolution/add pairs in each INT16 `DepthConvBlock` now use a fused
MMA epilogue. The convolution still rounds, adds bias, and clips to INT16 **before**
adding the residual and clipping again. Combining the two clamps would be wrong
for saturated convolution results followed by opposite-signed residuals.

The fused kernel supports 16-, 32-, and 64-column spatial tiles. The reduction
dimension, byte decomposition, accumulator wraparound, rounding, and clipping
are unchanged. Optional `DCVC_INT16_AUTOTUNE=1` selects a tile using median CUDA
event timing and caches the choice per device and shape in the current process.
Default execution uses 32 columns, which won most measured cases on this Jetson.
Uncached shapes use the default during graph capture. Tuning does not alter
checkpoints or prepared INT16 state. Older extensions and pre-Ampere GPUs retain
the separate convolution and residual operations.

## GPU profile and throughput

CUPTI profiling now succeeds from the root tmux shell authorized by the user.
The normal user still receives `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES` despite
`RmProfilingAdminOnly: 0` and membership in the device's `debug` group.

The steady 176x96 P-frame trace changed from 242 to 194 GPU kernels. There are
48 fused convolution/residual kernels and five remaining standalone residual
adds, compared with 53 standalone adds before fusion. This eliminates 48 kernel
launches and intermediate tensors per steady frame. Summed kernel duration in
the individual profiled frames changed from 21.767 to 21.341 ms. These profile
durations include profiling effects and are distinct from ordinary throughput.

Three paired, sequential codec runs used one model worker, 160 synthetic 176x96
frames, QP I/P 35/14, one I-frame followed by P-frames, and reset interval 32.
The first eight frames were excluded from steady throughput. Speed runs omit
the correctness runs' CPU feature hashing; every resulting stream was compared
byte-for-byte with the saved baseline from commit `4df627e`.

| Run | Baseline steady fps | Fused steady fps |
| --- | ---: | ---: |
| 1 | 74.035 | 80.092 |
| 2 | 69.136 | 81.999 |
| 3 | 69.914 | 76.987 |
| Median | 69.914 | 80.092 |

The median improvement is 14.6% on this Jetson Orin Nano. These are synthetic
codec measurements, not an estimate for the full download/audio/FFmpeg pipeline
or another GPU. GPU clocks were not locked; all three pairs favored fusion.

## Correctness

- 93 residual/tile cases cover full-range values, overflow, opposite-signed
  saturated residuals, odd shapes, multiple batches, noncontiguous tensors, and
  a non-default CUDA stream.
- All 138 saved convolution cases and the exhaustive 65,536-value rounding check
  pass with the new native extension.
- A standalone compute_75 PTX build passed the residual kernel's scalar fallback
  for all three tiles after JIT compilation on Orin. A physical pre-Ampere GPU
  was not available for validation.
- Default and autotuned 160-frame 176x96 runs match the baseline bitstream,
  reference features, every decoded frame, and combined decoded-pixel hash.
- Autotuned 40-frame 320x180 (QP I/P 0/45) and 128x72 (QP I/P 63/0) runs also
  match all those outputs, with intra interval 16, reset interval 8, and two
  entropy coders. These exercise padding, periodic I-frames, and different QPs.
- All 47 existing unit tests pass.
- Uncached-shape CUDA Graph capture and replay with changed residuals pass;
  ordinary autotuning populates and reuses the shape cache.
- The full FFmpeg/prefetch/conversion/encode/atomic-write pipeline produces an
  identical 159-frame stream with the old and new extensions. The fixture's
  frame-rate filter emits 159 frames for this 160-frame input. A single pair
  measured 52.4 versus 61.0 fps; the repeated codec measurements above provide
  stronger timing evidence.

Run the tile and residual regression benchmark after building the extension:

```bash
python benchmarks/benchmark_int16_residual.py \
  --extension /path/to/inference_extensions_cuda.so --output /tmp/residual_tiles.json
```

## CUDA Graph evaluation

Full steady P-frame capture after warmup fails at `entropy_models.py`'s
`symbols.to(torch.int8).cpu().numpy()` in `encode_z`, with
`operation not permitted when stream is capturing`.

A graph containing only feature extraction captured successfully and matched
eager output for the current feature, zeros, and INT16-minimum features. An
exploratory 100-iteration measurement gave 3.343 ms eager versus 3.062 ms replay
for that stage. Production graph capture remains unenabled: integration requires
separating GPU computation from entropy coding and managing graph-owned buffers
across reference updates, QP changes, resets, and resolutions. This evaluation
identifies a viable partial capture boundary; it does not provide full-frame
graph support.

Detailed local evidence is in `benchmarks/optimization_20260907/` on the test
machine. Generated streams, models, native binaries, and profiler traces are
excluded from the commit.
