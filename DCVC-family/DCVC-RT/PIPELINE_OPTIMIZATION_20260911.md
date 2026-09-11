# Frame preparation, transfer batching, and remaining GPU graph experiments

Baseline: commit `d01301c`, including the feature/main/hyper-encoder CUDA Graph.
Only the frame-input optimization was retained in normal runtime code. The
transfer and larger-graph prototypes are stored as explicit experimental patches
for future testing on other GPUs; updating the repository does not activate them.

## Selected change

Set `DCVC_INT16_FAST_INPUT=1` alongside `DCVC_USE_INT16=1` for `encode` or `stream`.
The flag is optional and off by default. It does not require rebuilding native
extensions, changing prepared models, or changing decoder arguments.

The original path uses SciPy to expand chroma, assembles YUV444, normalizes pixels
in float32, casts to float16, pads the image, and subsequently converts features
to INT16 inside the model. The new path builds a 256-entry lookup table on the
selected GPU using exactly that float32/float16/INT16 conversion. CPU lookups then
produce the same INT16 values directly in a reusable pinned YUV444 buffer.
Nearest-neighbor chroma replication and edge padding are filled into that buffer
without general interpolation or per-frame GPU normalization kernels.

One device buffer is reused per video. An event fences the previous upload before
the CPU rewrites the pinned host buffer. Device consumers and subsequent uploads
are ordered on the same CUDA stream. The buffer is drained during cleanup.
The converter is scoped to one video/stream; concurrent workers own separate
instances. Float and CPU encoding keep their existing input path.
Per-video lookup/buffer setup and graph recapture for the INT16 input dtype add
startup work. Very short clips may not amortize it; the speedup below is measured
on longer clips.

## Measurement and decision

A CPU profile of the old 159-frame pipeline attributed 0.108 seconds to YUV420
conversion (about 0.68 ms/frame). Entropy transfers also showed repeated blocking
CPU copies. Profiling timings include instrumentation and were used to select
experiments, not to calculate the speedup.

The performance benchmark uses a single persistent model worker, first warms it,
then rotates candidate order across three rounds. Checkpoints, prepared kernels,
autotuning choices, input frames, and coding settings are shared across variants.
Measurements include FFmpeg, frame preparation, encoding, and atomic output
writing. Audio and network downloading are excluded. CPU/GPU clocks were not
locked or changed. No elevated permissions were required.

Configuration: Jetson Orin Nano, PyTorch 2.5 / CUDA 12.6, INT16 autotuning enabled,
encoder CUDA Graph enabled, 176x96 synthetic video, 24 FPS, QP I/P 35/14, reset
interval 32. Longer clips losslessly repeat the earlier fixture. The 1,200 input
frames yielded 1,199 encoded frames per timed run.

| Run | Baseline FPS | Fast input FPS | Batched transfer FPS | Both FPS |
| --- | ---: | ---: | ---: | ---: |
| 1 | 82.813 | 87.086 | 82.190 | 86.776 |
| 2 | 82.285 | 87.089 | 81.818 | 87.650 |
| 3 | 82.553 | 86.334 | 81.234 | 87.079 |
| Median | 82.553 | 87.086 | 81.818 | 87.079 |

Fast input improved the median by **5.5%**, with all three pairs favoring it.
Earlier 400-frame screens also favored it. Peak allocated CUDA memory in all
twelve longer timed runs was 404,703,232 bytes; this is allocator usage with a
persistent warmed model, not total memory including pinned buffers.

The optimized transfer prototype queued Z/Y copies into reusable pinned buffers,
retained GPU source tensors until DMA completed, and waited once at flush before
passing arrays to the native entropy coder in their original order. The native
encoder copies arrays into owned vectors synchronously. Capacity was reused for
variable symbol counts, with explicit cancellation fencing at reset. A second
version removed unnecessary per-copy event recording. Although short tests
suggested a gain, the longer measurements showed a **0.9% regression alone** and
essentially no benefit alongside fast input. It was therefore not retained in
normal runtime code.

For the remaining GPU stages, two graph arrangements were tested:

| 400-frame comparison | Existing encoder graph FPS | Candidate FPS |
| --- | ---: | ---: |
| Single graph through prior computation and reference reconstruction | 79.023 | 71.055 |
| Separate later-stage graph, retaining the earlier Z readiness event | 79.444 | 72.613 |

These are medians from separate three-round screens. Both arrangements were also
tested with input/transfer optimizations and remained slower. Graph-owned
reference features were cloned before entering the DPB to prevent later replay
from overwriting retained references. Outputs were correct, but the performance
result rejected both arrangements. The exact cause of their slowdown was not
isolated; the existing encoder graph remains unchanged.

These results are measurements on this Jetson. The input code uses NumPy and
PyTorch CUDA operations rather than Jetson-specific APIs. Performance on discrete
GPUs, other resolutions, and different worker counts needs separate measurement.

## Correctness and reproduction

- The new input-buffer test covers all 256 sample values, changed frames, chroma
  replication, padded and unpadded shapes, a non-default CUDA stream, and reuse
  while GPU consumers are queued. Resulting INT16 tensors match the original
  conversion exactly at 256x2, 18x10, 176x96, and 320x180.
- Normal pipeline streams match byte-for-byte at 176x96, 320x180, and 128x72.
  A 159-frame new-input stream also matches the saved baseline from before this
  experiment. The twelve long runs match each other and the previous 1,199-frame
  saved stream.
- All 39 timed 400-frame candidate runs in the two screens matched their respective
  baseline streams. The initial 40-frame transfer and full-graph checks also matched
  encoder reference hashes and all decoded frames.
- All 84 unit tests passed, including the new CUDA input-buffer regression.

Run the retained comparison from the project directory with a sufficiently long
synthetic fixture and a fresh output directory:

```bash
DCVC_INT16_AUTOTUNE=1 python benchmarks/benchmark_encode_options.py --fixture /path/to/fixture.mkv --frames 1200 --rounds 3 --output /tmp/input-comparison
```

The benchmark enables INT16, uses the existing encoder graph, and selects the
baseline/input candidates by default. It never cleans up source media.

To investigate rejected candidates on another GPU, use a disposable checkout and
apply the desired patch **from the Git repository root**:

```bash
git apply DCVC-family/DCVC-RT/benchmarks/experiments/async_entropy.patch
git apply DCVC-family/DCVC-RT/benchmarks/experiments/graph_boundaries.patch
```

Both patches are based on the unchanged production model sources at `d01301c`.
Then run the benchmark from the project directory with, for example,
`--cases baseline input async input_async full full_input full_async all stages stages_input_async`.
The benchmark rejects experimental cases when the required patch is absent.
These prototypes are not production recommendations or measured wins on another
GPU. Only the existing graph and optional fast input are active production paths.

Generated evidence remains local under `benchmarks/optimization_20260911_followup/`.
No models, compiled extensions, test videos, captures, or generated streams are
included in the commit.
