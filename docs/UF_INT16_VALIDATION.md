# UF integer validation on Jetson Orin

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
