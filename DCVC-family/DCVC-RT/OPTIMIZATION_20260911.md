# Partial CUDA Graph encoding experiment

## Implementation

`DCVC_INT16_CUDA_GRAPH=1` enables CUDA Graph replay for the P-frame feature
extractor, main encoder, hyper-encoder, and hyperlatent rounding during INT16
inference. It is disabled by default and applies to both local encoding and remote
streaming because they use the same `DMC.compress`.
It requires the existing working INT16 CUDA extension, with no native rebuild.
`DCVC_INT16_CUDA_GRAPH=feature` retains the initial feature-only boundary for
comparison. Setting `0` disables capture.

Only this pure GPU stage is captured. Reference adaptation, prior decoding,
entropy coding, reconstruction of reference features, and reference updates still
execute normally. The CPU entropy-symbol transfers that prevented full-frame
capture are outside the graph.

The graph owns fixed input/output buffers. Every replay copies the current adapted
reference feature, input frame, and both QP scales into the input buffers, so
frame-level QP changes and I-frame/reset transitions cannot reuse stale inputs.
Decoder-generated DPB features are allocated outside the graph. GPU consumers run on the calling CUDA
stream. The entropy stream's blocking CPU copy of graph-owned Z symbols finishes
before `compress` returns, so it also completes before replay or invalidation.

Each model retains only one shape/device/stream entry. Changing that signature
finishes outstanding replay-stream work, releases the previous entry, warms the
kernels and optional autotuning, and captures a replacement. Model movement,
state-dict loading, and INT16 preparation also invalidate the graph. Runtime
models are assumed to have fixed prepared weights between these operations.
Training, gradient-enabled calls, float execution, and decoding retain the eager
path. An enclosing graph capture also bypasses the internal graph cache.

There is a one-time warmup/capture cost for each new signature and additional GPU
buffer memory. Short videos and frequently changing sizes may not amortize it.
Use one model per worker, as the CLI already does; the graph cache is not intended
for simultaneous calls to the same model from several threads.

## Initial feature-only measurements

These measurements were taken before expanding the graph boundary. They describe
the mode now selected by `DCVC_INT16_CUDA_GRAPH=feature`.

Jetson Orin Nano, existing PyTorch 2.5 / CUDA 12.6 environment, one encoding
worker, `DCVC_INT16_AUTOTUNE=1`. Three paired runs alternated execution order.
CPU and GPU clocks were not locked or changed; CPU governor was `schedutil`.
All runs used the local synthetic 176x96 fixture, QP I/P 35/14, and reset interval
32. These results do not predict performance on other GPUs or over the network.

| Run | Eager steady codec FPS | Graph steady codec FPS | Eager full pipeline FPS | Graph full pipeline FPS |
| --- | ---: | ---: | ---: | ---: |
| 1 | 76.191 | 86.963 | 59.178 | 60.807 |
| 2 | 80.677 | 85.140 | 55.096 | 62.660 |
| 3 | 77.487 | 83.262 | 60.873 | 60.799 |
| Median | 77.487 | 85.140 | 59.178 | 60.807 |

The codec-only median improved **9.9%**, with all three pairs favoring graph replay.
These runs encode 160 frames, exclude the first eight from steady timing, and
omit per-frame CPU hashing. Every output stream still matched byte-for-byte.

The full FFmpeg/prefetch/conversion/encode/atomic-write pipeline median improved
**2.8%**. This includes first-frame setup and graph capture. The frame-rate filter
emits 159 frames for the fixture. One pair was essentially tied (graph was 0.12%
slower), so the smaller full-pipeline gain is less conclusive than the codec-only
result. Audio and YouTube downloading are not included in this measurement.

Peak allocated CUDA memory in the codec speed runs was 420,190,720 bytes eager
and 420,327,424 bytes with the graph. This is allocator peak usage for this small
fixture, not the total reserved graph memory or a bound for larger resolutions.

## Expanded graph comparison

The expanded boundary was then compared with both eager and feature-only execution
under the same settings. Three rounds rotated the order of all three modes; the
models ran sequentially, one GPU worker at a time. Clocks remained unlocked.

| Run | Eager codec FPS | Feature graph codec FPS | Expanded graph codec FPS | Eager pipeline FPS | Feature graph pipeline FPS | Expanded graph pipeline FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 79.543 | 87.062 | 87.659 | 61.012 | 61.940 | 63.807 |
| 2 | 74.703 | 86.972 | 87.421 | 52.900 | 58.474 | 63.215 |
| 3 | 84.469 | 86.277 | 87.477 | 58.546 | 53.394 | 60.536 |
| Median | 79.543 | 86.972 | 87.477 | 58.546 | 58.474 | 63.215 |

Expanded capture improves the codec median by **0.6% over feature-only capture**
in this comparison. All three pairs favor expansion, but the incremental benefit
is small. The 159-frame pipeline medians differ by 8.1%; substantial variability
in these short runs prompted the longer pipeline comparison below rather than
treating that percentage as a reliable overall speedup.

Peak allocated CUDA memory was 420,164,608 bytes eager, 420,301,312 bytes for
feature-only capture, and 420,403,200 bytes for expanded capture in all three codec
rounds. All nine codec streams match the saved eager baseline byte-for-byte; the
nine pipeline streams also match their saved eager baseline.

The initial 40-frame expanded test printed two `NvMapMemAllocInternalTagged`
allocation warnings, but finished with exact bitstream, feature, and decoded-pixel
matches. The stage regressions and all 18 subsequent short performance runs
completed without those warnings. No memory limits, clocks, or governors were
changed in response.

The longer comparison used 1,200 input frames made by losslessly repeating the
same fixture. FFmpeg emitted 1,199 encoded frames per run. Each run includes
capture setup, frame preparation, and atomic output writing. Three pairs
alternated feature-only and expanded execution order:

| Run | Feature graph pipeline FPS | Expanded graph pipeline FPS |
| --- | ---: | ---: |
| 1 | 71.986 | 80.055 |
| 2 | 80.133 | 80.758 |
| 3 | 79.545 | 80.828 |
| Median | 79.545 | 80.758 |

This gives an additional **1.5% median full-pipeline improvement over feature-only
capture**. Every pair favored expansion, but the first feature-only run was
considerably slower than the others. Together with the 0.6% codec improvement,
these results support a modest incremental benefit on this Jetson, not the 8.1%
suggested by the short pipeline runs. All six long streams were byte-identical,
and none printed allocation warnings. Other GPUs and resolutions remain unmeasured
for performance. The flag stays optional and disabled by default.

## Correctness and reproduction

- Forty-frame eager/graph runs at 176x96, QP I/P 35/14, reset 32 matched bitstreams,
  all reference feature hashes, every decoded frame, and the combined pixel hash.
- Forty-frame runs at 320x180 (QP 0/45) and 128x72 (QP 63/0), intra interval 16,
  reset interval 8, and two entropy coders matched the same outputs.
  Both graph boundaries were checked against the saved eager results at all three
  resolutions.
- The stage regression exercises 120 exact cases across both graph boundaries:
  zero, INT16 minimum/maximum,
  random features, QPs 0/14/22/63/71, multiple shapes, and default/side CUDA streams.
  Expanded capture also receives a fresh input frame and encoder QP scale each time.
  All five outputs (both contexts, main latent, rounded hyperlatent, and hyperlatent
  symbols) are checked. QP changes reuse the same graph. Model movement, loading,
  and preparation clear it.
- Twenty-four additional P-frames per mode reuse one model across four video sizes
  with resets and changing QPs, comparing eager, feature-only, and expanded graph
  streams and reference features.
- All six codec speed outputs and all six full-pipeline outputs match their
  respective eager streams byte-for-byte. Bitstream syntax and naming are unchanged.
- All 83 existing unit tests passed.

Run the direct replay and persistent-worker checks:

```bash
DCVC_INT16_AUTOTUNE=1 python benchmarks/check_feature_graph.py
```

For paired codec checks, use fresh output directories and the same synthetic
fixture. The benchmark enables INT16 itself:

```bash
DCVC_INT16_AUTOTUNE=1 DCVC_INT16_CUDA_GRAPH=0 python benchmarks/benchmark_synthetic_codec.py --fixture /path/to/fixture.mkv --frames 160 --output /tmp/codec-eager
DCVC_INT16_AUTOTUNE=1 DCVC_INT16_CUDA_GRAPH=1 python benchmarks/benchmark_synthetic_codec.py --fixture /path/to/fixture.mkv --frames 160 --output /tmp/codec-graph --compare /tmp/codec-eager
```

Add `--speed-only` to both commands for throughput measurements without per-frame
CPU hashing. Replace the second command's setting with `DCVC_INT16_CUDA_GRAPH=feature`
to measure the older boundary. Full-pipeline measurements use
`benchmarks/benchmark_codec_pipeline.py` with the same environment switch and the
installed native `--extension` path. The long runs used `--frames 1200`; generate a
long enough fixture before running them.

Local generated evidence is stored under `benchmarks/optimization_20260911/`.
Models, binaries, test videos, and generated streams remain outside version control.
