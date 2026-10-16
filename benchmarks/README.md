# UF INT16 HT-L calibration benchmark

[uf-int16-htl-144p-pareto.json](uf-int16-htl-144p-pareto.json) is the complete,
unchanged result of our earlier 2,000-trial search, completed September 13, 2026
UTC on the Jetson. All 2,000 trials succeeded. The published copy lives outside
the ignored `runs/` directory and ships with the code; models and test media do
not. Original local paths in the metadata record provenance and are not needed
to use the results.

The source was the 10-second, 300-frame Big Buck Bunny sample, resized from
1280×720 to **256×144**, at 30 FPS. Source SHA-256, prepared-model identities,
search seed, parameter domains and measurements are retained in the JSON.
PSNR is mean per-frame YUV420 PSNR with 6:1:1 Y:U:V weighting. Byte sizes are
video bitstreams only, excluding audio and metadata. Encoding seconds are the
codec's measured encoding duration, excluding decoding and download startup.

`encode`/`stream --target-psnr` searches every successful trial with measured
PSNR at least the target, choosing minimum encoding time, then minimum bytes
for a time tie, then trial number. Restricting selection to the 2D size/PSNR
frontier would discard potentially faster choices. This is a speed-first
selection, not simultaneous independent minimization of size and time.
The selected codec parameters can be used with any explicit `--resolution`;
without one, the selector defaults to the benchmark's 144p short edge.
The recorded PSNR, speed and sizes remain measurements at 256×144.

For `--target-psnr 30`, trial **783** is selected:

| QI | QP | Reset interval | Intra period | Skip threshold | PSNR (dB) | Encode seconds | Bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 14 | 2 | -1 | 0.2 | 31.225371 | 3.992191 | 17,830 |

These are reference measurements, not a promise that arbitrary videos reach
the same PSNR or encode at the same speed. The selector does not retrain models,
measure input quality, or run an extra encode/decode pass.

## Local Python throughput

`benchmark_python_uf.py` measures the INT16 Python path with FFmpeg input,
bounded frame prefetch and bitstream writes. It uses existing prepared UF
checkpoints and never downloads media or models. Generate a moving synthetic
video and run one and two synchronized workers:

```bash
mkdir -p runs/python-speed
ffmpeg -v error -f lavfi -i testsrc2=size=256x144:rate=30 -frames:v 257 -c:v ffv1 -threads 1 runs/python-speed/synthetic.mkv
./scripts/uf-python benchmarks/benchmark_python_uf.py --input runs/python-speed/synthetic.mkv --output runs/python-speed/measured --verify
```

The output directory must be new. The default settings are HT-L, 144p,
QI10/QP14, reset interval 2, skip threshold 0.2, three measured repetitions
and a 17-frame warmup. `--procs 1 2` selects worker counts; `--variant`,
`--resolution`, `--fps` and codec options permit other workloads.
`results.json` records each worker's FPS, startup time, peak RSS, stream hashes
and optional decoded pixel hashes. Aggregate pipeline FPS divides total frames
by the elapsed span across all workers, including FFmpeg input startup and
shutdown. Model initialization and verification decoding are excluded from
that number and reported separately. Two workers share the GPU, so aggregate
throughput need not exceed one worker's throughput.

The default `--gpu-scheduling auto` follows the archive CLI: INT16 HT-L workers
sharing a GPU coordinate complete encode chunks for frames at least 144p.
CPU input work remains concurrent. HT-S, LD and smaller frames keep concurrent
GPU execution. Use `--gpu-scheduling concurrent` to measure without chunk
coordination. Saved `--model-source` controls retain their original scheduling.
The report records the scheduling used by each worker.
For a saved model that already supports the GPU gate, add
`--reference-gpu-gate` to coordinate its encoding too. The report records
encoding and decoding coordination separately. Current HT-L workers also
coordinate decode chunks; older saved models keep their own decode behavior.

With `--verify`, a final worker barrier keeps verification outside every
worker's measured encode interval. `aggregate_verification_fps` separately
measures all verification decodes across their combined elapsed span.

`--verify` decodes in memory, hashes each frame and discards its pixels; no raw
YUV or other decoded media is written. Cross-decoding checks also keep only
pixel hashes. For the normal archive CLI, use `main.py verify --input_file
ARCHIVE.uf.json` for the same bounded verification without an output video.

`--model-source /path/to/saved/model.py` runs an earlier saved Python model
implementation with the same benchmark. Add `--layers-source` and
`--native-library` to restore the saved layer dispatcher and native extension
as well. The report records source and native-library SHA-256 hashes, so a
control run can prove that it used the previous implementation.
Use `--integer-codec-source` to restore its checkpoint initialization and
source conversion, `--ops-source` for its arithmetic dispatcher, and
`--entropy-library` for its saved `MLCodec_extensions_cpp` library. Worker
reports include the entropy-library hash separately from the CUDA-library hash.
Use `--archive-source /path/to/saved/archive/codec.py` to also restore the
original shared encode/decode wrapper, including its pixel validation. The
report records the loaded model, layer and archive implementation modules.
`check_python_uf_compat.py` compares
that saved implementation with the current model, requiring identical streams
and decoded pixel hashes in both decoding directions. It covers skip on/off,
frequent resets, repeated I-frames, partial final chunks and changed dimensions:

```bash
./scripts/uf-python benchmarks/check_python_uf_compat.py --input runs/python-speed/synthetic.mkv --reference-model /path/to/saved/model.py --output runs/python-speed/compat-htl --variant htl
```

Run separate invocations with `--variant hts` and `--variant ld` to cover the
other UF topologies without holding multiple codec models in memory. Use
`--reference-layers` and `--reference-native` with saved files to cross-decode
against the complete older implementation, including its original CUDA kernels.
