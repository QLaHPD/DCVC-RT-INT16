# Partial CUDA Graph encoding experiment

## Implementation

`DCVC_INT16_CUDA_GRAPH=1` enables CUDA Graph replay for the P-frame feature
extractor during INT16 inference. It is disabled by default and applies to both
local encoding and remote streaming because they use the same `DMC.compress`.
It requires the existing working INT16 CUDA extension, with no native rebuild.

Only the pure feature-extraction stage is captured. Reference adaptation, latent
encoding, entropy coding, and reference updates still execute normally. The CPU
entropy-symbol transfers that prevented full-frame capture are outside the graph.

The graph owns fixed input/output buffers. Every replay copies the current adapted
reference feature and QP scale into the input buffers, so frame-level QP changes
and I-frame/reset transitions cannot reuse stale inputs. Decoder-generated DPB
features are allocated outside the graph. All graph outputs are consumed on the
calling CUDA stream before the next replay.

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

## Measurements

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

## Correctness and reproduction

- Forty-frame eager/graph runs at 176x96, QP I/P 35/14, reset 32 matched bitstreams,
  all reference feature hashes, every decoded frame, and the combined pixel hash.
- Forty-frame runs at 320x180 (QP 0/45) and 128x72 (QP 63/0), intra interval 16,
  reset interval 8, and two entropy coders matched the same outputs.
- The feature regression exercises 60 exact cases: zero, INT16 minimum/maximum,
  random features, QPs 0/14/22/63/71, multiple shapes, and default/side CUDA streams.
  QP changes reuse the same graph. Model movement, loading, and preparation clear it.
- Twenty-four additional P-frames reuse one model across four video sizes with
  resets and changing QPs, comparing graph/eager streams and reference features.
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
CPU hashing. Full-pipeline measurements use `benchmarks/benchmark_codec_pipeline.py`
with the same environment switch and the installed native `--extension` path.

Local generated evidence is stored under `benchmarks/optimization_20260911/`.
Models, binaries, test videos, and generated streams remain outside version control.
