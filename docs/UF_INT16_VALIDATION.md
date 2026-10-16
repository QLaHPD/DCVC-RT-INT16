# UF integer validation on Jetson Orin

## Python convolution dispatch and integer boundaries — 2026-10-16

This round compares against the preceding local `packed-v5` implementation,
saved with its source and native library in
`runs/python-speed-20261016-round4/baseline/`. It measures further gains on top
of the preceding optimization round.

Common pointwise projections now fetch two 32-element reduction fragments
together. Shape selection uses the measured 32/64-column tiles and a two-warp
block for small square projections. The four-channel activation reduction
retains the smaller register set where that is faster. Rejected kernel trials
and per-shape timings are retained in the local run directory. The existing
16 MiB input-packing and 8 MiB receptive-field limits remain in force.

Symbol quantization, symbol restoration and reciprocal calculation each use
one integer CUDA pass. Multiplication has scalar, full-tensor and per-channel
broadcast paths that avoid general index calculations. Each INT16 product and
its rounding offset fit INT32; convolution dot products still accumulate
exactly into INT64. All original rounding and saturation boundaries remain.
CPU tensors, other input dtypes and older native libraries retain their
previous operations.

Canonical depth/residual blocks can dispatch their five convolutions through
one native call, using references to the existing prepared weights. This
reduces Python dispatch; the GPU kernels and intermediate saturation points
remain separate. An isolated HT-L comparison measured 111.32 versus
111.36 FPS with the older Python dispatcher, so that change alone has no
detectable HT-L throughput benefit in this workload.

The native revision is `dispatch-v6`. The existing extension was rebuilt with
`MAX_JOBS=1` and `TORCH_CUDA_ARCH_LIST=8.7`. No dependencies, additional model
copies, arithmetic identities, prepared identities or bitstream fields were
added. GPU scheduling uses the preceding implementation's per-invocation
HT-L gate.

The same Orin Nano at 1.02 GHz, local moving FFV1 input, HT-L INT16,
QI10/QP14, reset 2, intra -1, skip 0.2 and 30 source FPS were used. Values
are medians of three repetitions after a 17-frame warmup. Timing includes
FFmpeg input and bitstream writes; initialization and verification are
separate. Both workers finish timed encoding before either starts decoding.

| Dimensions | Frames per worker | Workers | Previous FPS | New FPS | Gain |
|---|---:|---:|---:|---:|---:|
| 256×144 | 257 | 1 | 100.27 | 111.32 | 11.0% |
| 256×144 | 257 | 2, aggregate | 101.48 | 112.69 | 11.0% |
| 512×288 | 129 | 1 | 27.65 | 29.47 | 6.6% |
| 512×288 | 129 | 2, aggregate | 27.96 | 29.80 | 6.6% |

The sustained two-worker run used 1,025 frames per worker and three measured
repetitions. Aggregate throughput improved 104.74 → 116.40 FPS (11.1%),
again with identical streams and pixel hashes. Measured peak RSS was about
1.20 GiB per worker at 144p and 1.24 GiB at 288p. Verification throughput
improved 89.64 → 96.23 FPS at 144p and 25.17 → 26.50 FPS at 288p with one
worker; two-worker verification improved 81.07 → 87.51 and 22.79 → 24.02 FPS.

The same settings with 129 frames per worker also checked the other UF modes:

| Mode | Workers | Previous FPS | New FPS | Gain |
|---|---:|---:|---:|---:|
| HT-S | 1 | 123.96 | 134.29 | 8.3% |
| HT-S | 2, aggregate | 121.29 | 134.69 | 11.0% |
| LD | 1 | 66.01 | 75.28 | 14.0% |
| LD | 2, aggregate | 74.45 | 77.75 | 4.4% |

All measured streams and decoded pixel hashes match the saved implementation.
All 125 regression tests and all 12 HT-L/HT-S/LD cross-decoding cases passed.
The checks include full INT16 conversion ranges, every INT8 symbol value,
multiply broadcasts, noncontiguous views, reduction tails, maximal accumulation
and each residual saturation boundary. The normal CLI also encoded, verified
and resumed two local clips with `--fps 4 --procs 2` (34 and 137 frames),
matching the previous decoded hashes. Original synthetic inputs were retained.
Verification hashes decoded frames in memory and writes no decoded media.
Reports, native/source fingerprints and the completion audit are under
`runs/python-speed-20261016-round4/`. These synthetic measurements exclude
network throughput and model initialization.

## Python packed matrices and residual fusion — 2026-10-16

This round compares against published commit `878742e`, including its saved
Python model, layer dispatcher and `ffn-v4` CUDA library. Large pointwise
convolutions can pack INT16 input into channel-last order once and read matrix
fragments directly, avoiding repeated shared-memory transposes and block
barriers. Measured shape rules retain the shared-memory path for smaller
matrices and large square matrices where packing loses time. Each channel-last
packing buffer is bounded to 16 MiB; spatial receptive-field packing retains
its existing 8 MiB bound.

Activation lookup and residual additions now run in the convolution epilogue.
Convolution rounding/saturation and each residual saturation remain separate,
in their original order. A specialized 3×3 depthwise kernel avoids the general
grouped-convolution loops. Reconstruction is queued before waiting for the
existing CPU rANS worker, allowing those operations to overlap.

HT-L workers sharing a GPU now coordinate complete encode chunks within their
invocation. Both input workers remain active. This applies to INT16 frames
with a short edge of at least 144 pixels; LD and HT-S retain concurrent GPU
execution. Each invocation has its own temporary lock directory and a separate
lock per shared GPU. Locks are released when a worker exits, including abnormal
termination. Scheduling paths are absent from archive identities and metadata.

Verification also has a fused integer conversion to cropped YUV420 bytes,
preserving luma rounding and chroma averaging/flooring. Integer pixels require
no finite-value reduction. Decoded frames are hashed in memory and discarded;
verification writes no decoded media. CPU inference and older native libraries
retain the original operations. Prepared weights, entropy tables, arithmetic
identity and bitstream format are unchanged; no additional models or libraries
are required.

The native kernel revision is `packed-v5`. Rebuild the existing extension with
`MAX_JOBS=1` using the command below. The complete regression suite passed all
119 tests, including independent CPU comparisons for packed matrix tails,
batching, maximal accumulation/cancellation, residual saturation order,
noncontiguous views, two-byte storage offsets, depthwise borders and every
INT16 pixel value. CPU-only process checks cover archive/stream coordination,
namespace cleanup, worker death and GPU completion before releasing a slot.

Measured on the same Orin Nano at 1.02 GHz, with HT-L INT16, QI10/QP14,
reset 2, intra -1 and skip 0.2. Local moving FFV1 input uses 30 source FPS.
Values are medians of three repetitions after a 17-frame warmup. Timing
includes FFmpeg input and bitstream writes; initialization and verification
are excluded. A worker barrier now prevents verification from overlapping
another worker's final encode. The control restores the original archive
wrapper as well as its model, layers and CUDA library.

| Dimensions | Frames per worker | Workers | Previous FPS | New FPS | Gain |
|---|---:|---:|---:|---:|---:|
| 256×144 | 257 | 1 | 84.84 | 100.45 | 18.4% |
| 256×144 | 257 | 2, aggregate | 78.01 | 101.70 | 30.4% |
| 512×288 | 129 | 1 | 22.48 | 27.76 | 23.5% |
| 512×288 | 129 | 2, aggregate | 20.37 | 27.96 | 37.3% |

Without chunk coordination, the new kernels measured 92.72 aggregate FPS at
144p and 25.29 at 288p. Final peak RSS was approximately 1.22 GiB per worker
at 144p and 1.27 GiB at 288p. The original control and optimized decoder
verified 69.56 → 89.31 FPS at 144p and 20.15 → 25.22 FPS at 288p with one
worker; aggregate two-worker verification improved 65.13 → 81.03 and
18.24 → 22.76 FPS respectively.

The sustained two-worker check used 1,025 frames per worker and three measured
repetitions with the same verification barrier. Aggregate throughput improved
80.03 → 104.91 FPS (31.1%), with identical streams and decoded hashes.

The final implementation passed all 12 HT-L/HT-S/LD cross-decoding cases using
noisy synthetic input and the saved older native library in both directions.
Measured streams and pixel hashes match. The normal CLI also encoded and
verified two local files with `--fps 4 --procs 2` (34 and 137 frames), retained
both originals, and resumed both archives without re-encoding. Full local
profiles, candidates, frozen baselines, hashes and measurements are under
`runs/python-speed-20261016-round3/`. Real channel throughput also depends on
source decoding and network throughput.

## Python integer convolutions — 2026-10-16

Further optimization of the Python INT16 runtime combines pointwise convolution
with WSiLU and its four-channel reduction. Convolution rounding and saturation
still happen before each activation lookup, followed by the original sum and
saturation. Small spatial convolutions now materialize their receptive fields
once before the integer matrix kernel, with temporary storage capped at 8 MiB.
Pointwise tile selection was also adjusted using measured kernel timings.
Prepared weights, CDFs, arithmetic identity and bitstream format are unchanged.
CPU inference and older native extensions retain the original operations.

Same Orin Nano and 1.02 GHz GPU, same local moving FFV1 synthetic input and
HT-L INT16 settings as the preceding optimization round: QI10/QP14, reset 2, intra -1,
skip 0.2 and 30 source FPS. Each value is the median of three runs after a
17-frame warmup, including FFmpeg input and bitstream writes and excluding
model initialization and verification. The larger test resizes that local
source to 512×288. Two workers share the GPU and process separate copies.

| Dimensions | Frames per worker | Workers | Previous FPS | New FPS | Gain |
|---|---:|---:|---:|---:|---:|
| 256×144 | 257 | 1 | 80.64 | 84.64 | 4.9% |
| 256×144 | 257 | 2, aggregate | 74.28 | 77.98 | 5.0% |
| 512×288 | 129 | 1 | 20.74 | 22.47 | 8.3% |
| 512×288 | 129 | 2, aggregate | 19.03 | 20.37 | 7.0% |

The 144p control restored the saved previous Python model, layer dispatcher
and native library, reproducing 80.62 FPS and 74.20 aggregate FPS. The 288p
baseline also used that complete previous implementation. Native library and
Python source hashes are recorded by the benchmark. The fused activation
alone measured 80.95 / 74.65 FPS; the spatial packing and tile selection account
for most of this round's improvement.

All measured bitstreams and decoded YUV hashes match the previous version.
The 12 HT-L/HT-S/LD cross-decoding cases additionally restore the older layer
dispatcher and CUDA library, and match the saved original fixture hashes in
both decoding directions. They cover skip off/on, resets, repeated I-frames,
partial chunks and padding at changed dimensions. Full-range CUDA arithmetic
is checked against the independent CPU reference, including partial tiles,
batching, ties, overflow bounds and saturation between the fused operations.
All 107 regression tests passed. Two-worker archive CLI encoding and resume
also passed with `--fps 4`; both archives contain 34 frames. The raw synthetic
CLI decode outputs were hashed and removed. Benchmarks and cross-decoding
checks decode in memory and retain only hashes; use `main.py verify` for
subsequent archive checks without writing decoded media.

Rebuild the existing `uf_int16_cuda` extension with `MAX_JOBS=1` after updating.
The native kernel revision is `ffn-v4`; its arithmetic version stays v1.
Reproduction commands and complete comparison options are in the
[benchmark instructions](../benchmarks/README.md#local-python-throughput).
Local profiles, tile trials, frozen baseline source/library, stream fixtures
and raw measurements are under `runs/python-speed-20261016/`.

## Python fused spatial partitions — 2026-10-15

The Python INT16 runtime now uses fused CUDA partition operations for symbol
quantization, entropy indexes and latent reconstruction. Encoding transfers all
spatial partitions together, then removes skipped symbols on the CPU. Decoding
transfers indexes and skip flags together. The arithmetic ID, prepared model
identities, rANS tables and bitstream format remain unchanged. CPU inference
and older extensions without these operations retain the original path.

Measured on the local Orin Nano with its GPU at 1.02 GHz, using a local FFV1
`testsrc2` synthetic video: 257 frames, 256×144, 30 source FPS, HT-L INT16,
QI10/QP14, reset interval 2, intra period -1 and skip threshold 0.2.
Medians below use three repetitions after a 17-frame warmup. FPS includes
FFmpeg input, bounded prefetch and bitstream writes; it excludes model loading
and verification decoding. Two workers process separate copies concurrently
on the same GPU; their aggregate FPS uses the elapsed span across both workers.

| Workers | Before (FPS) | Fused partitions (FPS) | Improvement |
|---:|---:|---:|---:|
| 1 | 72.98 | 80.31 | 10.0% |
| 2, aggregate | 69.46 | 74.21 | 6.8% |

Every measured stream is byte-for-byte identical to its saved pre-change
baseline (10,891 bytes), and decoded YUV SHA-256 hashes match. The 12 additional
cross-decoding cases cover HT-L/HT-S/LD, skip off/on, frequent resets, repeated
I-frames, partial final chunks and changed input dimensions. For every case,
the new decoder reads the original implementation's stream and the original
decoder reads the new stream with identical pixels. All 103 regression tests
passed, including full-range arithmetic and independent CPU checks of the
new partition kernels.

A separate check decoded the saved 257-frame pre-change file directly with the
new decoder and matched its original pixel hash. Forcing the original path
also reproduced the original bitstream and pixels through the older-extension
fallback. The normal archive CLI encoded and batch-decoded two local synthetic
videos with two workers and `--fps 4`; both outputs contained 34 frames and had
identical decoded hashes. Synthetic sources have no YouTube `info.json`; use
`--allow-missing-metadata --keep-originals` when checking their archive inventory.
The baseline control rerun reproduced 72.63 FPS (one worker) and 69.52 FPS
(two workers), consistent with the initial baseline above.

Rebuild the existing extension to enable the fast path:

```bash
MAX_JOBS=1 ./scripts/uf-python -m pip install --no-build-isolation ./src/int16/native
```

See [the reproducible synthetic benchmark](../benchmarks/README.md#local-python-throughput).
Local raw measurements, pre-change model source and compatibility fixtures are
under `runs/python-speed-20261015/`. These FPS figures describe this workload
and hardware; download/network throughput and other resolutions were not part
of this measurement.

## HT-L 144p indexing and CUDA tiles — 2026-09-12

Compared with `07ffd08` using the same initialized HT-L model, QI36/QP30,
256×144, 300 Bunny frames and one worker. Three runs per mode alternated;
each encoded bitstream matched the original baseline byte for byte.

| Measurement | Previous code | Cached positions and tiled kernels |
|---|---:|---:|
| Full input/encoding pipeline, median FPS | 55.61 | **65.91** |
| Preloaded input, median codec FPS | 64.50 | **72.23** |

The full pipeline improves by **18.5%** and the preloaded codec by **12.0%**.
Full-pipeline defaults use two decoder threads in the previous version and
four here on this six-CPU Jetson; both use eight-frame bounded prefetch.
The new default divides available CPUs among workers, reserves one CPU per
worker for neural dispatch, and limits input decoder threads to 1–4.
Explicit `--input_threads` overrides this. FP16 defaults remain unchanged.
Encoding FPS excludes model loading and the subsequent validation decode.

The prior caches ordered flat mask positions for one shape, replacing repeated
boolean indexing with indexed gathers. Pointwise CUDA integer convolutions use
64-element reduction tiles or wider spatial tiles selected by shape. Generic
and non-pointwise paths retain their original behavior. Arithmetic, prepared
identities, artifact naming and entropy-symbol order are unchanged. The final
300-frame decode matches the original YUV hash below.

A separate CLI run using the new defaults encoded all 300 frames at 64.09 FPS
and passed full validation with identical compressed bytes and decoded pixels.
All 23 unit tests passed, including both tile paths, odd reduction tails,
multi-batch tensors, accumulator limits, mask partition order and CPU budgets.
HT-L, HT-S and LD passed CPU/CUDA checks against their earlier reference
reports, including reset states and feature-only encoder parity.
A 17-frame 720p HT-L CLI run with resets and I-frames every eight frames also
encoded and validated with the exact earlier bitstream and reconstruction
hashes. It emitted Jetson NvMap allocation warnings but completed successfully;
this short test does not establish memory headroom for multiple workers.

Rebuild `src/int16/native` after updating; the setup checker reports `tiled-v2`.
Existing prepared model files are reused. The indexing and input changes are
portable; tile heuristics were measured only on this Jetson Orin. Other GPUs
need their own performance and cross-device consistency checks.

CUDA graph prototypes added cold-start cost and memory pressure for a small
warm-run gain. An exact four-way INT8 BLAS decomposition was substantially
slower than the custom kernel. Neither prototype is part of the runtime.

## Earlier HT-L 144p optimization (`07ffd08`) — 2026-09-12

Same Bunny source, INT16 HT-L, QI36/QP30, 256×144 at 30 FPS, all 300 frames,
one process. Prepared model and arithmetic identities are unchanged.

| Full input/encoding pipeline | Median encode FPS (3 runs) |
|---|---:|
| Previous behavior: full P reconstruction, one input thread, no prefetch | 29.44 |
| Optimized defaults: required features/reset only, two input threads, bounded prefetch | **55.95** |

This is about **1.90× faster** in a six-run alternating comparison using the
same initialized model. Individual optimized runs measured 55.89–56.35 FPS.
Encoding FPS excludes model loading and the subsequent validation decode. The
earlier standalone CLI baseline measured 28.85 FPS; the optimized standalone CLI
measured 55.35 FPS. Four input threads with prefetch measured 57.79 FPS in one
separate run; that version used two by default to limit per-worker CPU usage.

Measurements separating the causes:

- With all input frames preloaded, skipping unused P-frame pixel reconstruction
  raised median codec throughput from 49.67 to 64.53 FPS (three runs per mode).
- Reading/resizing the input alone improved from 38.38 to 59.81 FPS with two
  decoder threads. Pixel hashes matched across tested decoder/filter settings.
- Bounded read-ahead then overlaps input work with GPU encoding. All generated
  full-video bitstreams match the original HT-L baseline exactly:
  `42e73b602cfc44fe5311224d119d043ed0bb6148d36b3021c5b581fa404cfbb1`.
- Full decoded YUV remains
  `5fca3d1b3961f2ca3678dc704f0e34e23c228dcb5f792c6cb32c90123de4444c`.

The default archive command enables these changes automatically:

```bash
./scripts/uf-python main.py encode --input_file Big_Buck_Bunny_720_10s_30MB.mp4 --output_root runs/optimized-htl --runtime int16 --model_structure htl --resolution 144 --qi 36 --qp 30 --procs 1
```

The arithmetic checker now compares feature-only encoding against full
reconstruction, including encoded bytes and reference/memory/context at resets.
Input tests cover frame order/content, repeated EOF, decoder failure, closing a
full prefetch queue, and resuming with changed performance settings. The changes
use standard CPU threading and skip unnecessary neural work; they are not
Jetson-specific, although these throughput measurements are from Jetson Orin.

FP16 keeps its original input defaults (one decoder thread, no prefetch). A
720p FP16 test with the larger input buffers exhausted this Jetson's memory;
with the original settings, all 300 frames encoded and validated with the exact
original FP16 bitstream. Faster input defaults therefore apply automatically
only to INT16. Early input cancellation stops the owned read-only FFmpeg child
immediately, avoiding its SIGTERM flush delay when the pipe is full.

Final validation: all 20 unit tests passed; HT-S, HT-L and LD passed CPU/CUDA
state checks against pre-optimization reference reports. A partial 17-frame
720p HT-L archive with resets, periodic I-frames and a padded final chunk also
encoded and validated successfully. Fixing early shutdown preserved its exact
bytes and pixels while removing the observed 10-second input-close delay.
The Bunny source remained unchanged. No native rebuild or model re-preparation
is needed for these optimizations.

The following baseline sections describe code before these optimizations
(published at `a938b27`).

## HT-L 144p baseline — 2026-09-12

HT-L is the focus for subsequent optimization. This run used INT16 Tensor Core
inference, QI36/QP30, reset interval 32, one process and all 300 Bunny frames.
1280×720 was resized to 256×144, preserving 30 FPS. Prepared caches were reused.

| Measurement | HT-L | Earlier HT-S baseline |
|---|---:|---:|
| Encoding loop | **28.85 FPS** / 10.400 s | 32.23 FPS / 9.308 s |
| Validation decode | **55.54 FPS** / 5.401 s | 76.43 FPS / 3.925 s |
| Bitstream | 28,583 bytes | 28,293 bytes |
| Frames / packets | 300 / 39 | 300 / 39 |

All frames passed validation. These are separate single-run stage timings;
encoding FPS excludes model loading/preparation and validation. No inference
changes were made for this benchmark.

```bash
./scripts/uf-python main.py encode --input_file Big_Buck_Bunny_720_10s_30MB.mp4 --output_root runs/uf-int16-htl-144p-baseline-20260912 --runtime int16 --model_structure htl --resolution 144 --qi 36 --qp 30 --procs 1
```

Validated raw YUV SHA256:
`5fca3d1b3961f2ca3678dc704f0e34e23c228dcb5f792c6cb32c90123de4444c`.

## 144p baseline — 2026-09-12

Current Tensor Core INT16 runtime, HT-S, QI36/QP30, reset interval 32,
one process, all 300 Bunny frames. Original 1280×720 resized to 256×144;
the source's 30 FPS was preserved. Existing prepared caches were reused.

| Measurement | Result |
|---|---:|
| Encoding loop | 9.308 s / **32.23 FPS** |
| Validation decode | 3.925 s / **76.43 FPS** |
| Bitstream | 28,293 bytes / 39 packets |
| Validation | All 300 frames passed |

These are single-run stage timings, excluding model loading/preparation from
encoding FPS and measuring validation separately. No inference changes were
made for this benchmark.

```bash
./scripts/uf-python main.py encode --input_file Big_Buck_Bunny_720_10s_30MB.mp4 --output_root runs/uf-int16-144p-baseline-20260912 --runtime int16 --model_structure hts --resolution 144 --qi 36 --qp 30 --procs 1
```

Code-inspection candidates for subsequent optimization:

1. Avoid reconstructing all P-frame pixels during encoding when only the
   reference features are needed; retain necessary reset-frame reconstruction.
   `IntegerModel.compress()` currently calls the full reconstruction path.
2. Capture suitable fixed-shape neural sections in CUDA graphs to reduce Python
   and launch overhead; keep entropy transfers outside capture where required.
3. Fuse prior masking, quantization and symbol/index packing, and reduce separate
   synchronous CPU transfers for the entropy coder.

These are candidates, not measured gains. Any change must preserve the existing
bitstream and reconstructed-state hashes.

## 720p validation

Test source: `Big_Buck_Bunny_720_10s_30MB.mp4`, 300 frames, 1280×720,
30 FPS. HT-S, QI36/QP30, reset interval 32, original resolution/FPS.
Source SHA256: `a2c61fa8bdb4381a28a06a4df53fdda35e35dd49082b99bf199e3125be38f3b9`.

| Measurement | FP16 baseline | INT16 generic | INT16 Tensor Core |
|---|---:|---:|---:|
| Bitstream bytes | 434,740 | 452,408 | 452,408 |
| Packets | 39 | 39 | 39 |
| Encode seconds | 11.418 | 330.610 | 85.819 |
| Average PSNR against source (dB) | 34.422331 | 34.253348 | 34.253348 |

The INT16 stream includes a 72-byte runtime/prepared-identity preamble. Its
roughly 4.1% larger size and 0.17 dB lower PSNR are measurements on this one clip,
not a general rate-distortion claim. Integerization changes the codec's numerical
behavior; it does not reproduce the floating model's exact reconstruction.

The Tensor Core decoder decoded the generic integer bitstream in 78.148 seconds
and produced exactly the same validated raw YUV hash:
`8326cf7aabe07ac4f5806cadb08ade4e79dce30373e506e70908f1573e1a1852`.
The exported FFV1 Matroska file was independently counted as 300 frames at 720p30.

The Tensor Core encoder also completed all 300 frames, including validation,
and produced a byte-identical bitstream to the generic encoder, SHA256:
`7b036185f86f7100c356dd9fc8f4b4f26152aa0917053e349159337c21e07cd0`.
Encoding is about 3.85× faster than the generic integer implementation, but
still substantially slower than the optimized FP16 baseline. These are separate
single-run measurements, not controlled statistical benchmarks.

Arithmetic tests cover independent scalar/CPU convolution, exact accumulation
beyond INT32, reduction lengths 32768 and 32769, cancellation, odd tiles,
unsaturated outputs, general/strided/grouped convolutions and fused integer
operations. Identity tests reject changed prepared contents, wrong source/model
identities, nonfinite checkpoint parameters and conflicting bitstream runtimes.
Archive tests cover atomic publication, interrupted validation, cooperative
claims, pipeline mismatch, resume and exact-path cleanup.

A separate real-model fixture passed folder encoding with Opus audio, periodic
I-frames, unchanged-resolution handling, resume without changing the bitstream
hash/mtime, CPU verification, forced-runtime rejection and live decoding through
ffplay's dummy SDL driver. Both encoding and decoding used explicit prepared
files with deliberately nonexistent floating-checkpoint paths. Cleanup preview
preserved the generated source; confirmed cleanup validated and removed only
that fixture. The user's Bunny source hash remained unchanged.

A two-frame Bunny archive encoded and validated on the CPU reference at 56×32
also passed fresh CUDA verification, including a padded final temporal chunk.
All 17 unit tests passed; the integer arithmetic subset was rerun successfully
after adding maximal low-byte accumulation at the Tensor Core reduction bound.

The HT-S consistency checker matched the pre-optimization bitstreams and all
reconstruction/reference/memory/context hashes on CUDA and the CPU reference,
including temporal resets. HT-L at 32×32 and LD at 48×48 also passed CUDA/CPU
bitstream reconstruction and temporal-state checks. The checker rejects
floating-point tensors during inference. Fresh FP16 verification still matches
the original 300-frame reconstruction hash:
`7287c2dc7c724900f215955f8d20bd378b22aefd32452be99c0232dba341689d`.
These tests ran in the dedicated UF environment with NVIDIA PyTorch
2.5.0a0+872d972e41.nv24.08 and CUDA 12.6. Other physical GPU models remain
unverified; run `scripts/check_uf_int16.py --compare` there with the same prepared
files. Native binaries must be rebuilt for the target environment.

Initial generic runs exhausted the 8 GB Jetson's shared RAM. Fused integer
elementwise kernels eliminated large intermediate tensors; the full generic
encode and validation then completed with one worker. Use `--procs 1` on this
machine. Test artifacts and models remain local and are excluded from Git.
