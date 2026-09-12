# UF integer validation on Jetson Orin

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
