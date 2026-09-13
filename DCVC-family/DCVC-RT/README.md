# DCVC-RT INT16 Managed Runtime

This subtree extends the official [DCVC-RT](UPSTREAM_README.md) implementation with deterministic INT16 execution, bit-exact CUDA performance work, a unified media CLI, and a fail-closed archival lifecycle.

It is intended for long-running, headless video compression where encoded video, Opus audio, and source metadata must remain independently manageable and where predictive decoding must behave consistently across supported devices.

## Main modifications

### Configurable model training

`main.py train --config configs/train/base.yaml` trains configurable image/I-frame
and video/P-frame models, including exact INT16 forward computation with
surrogate gradients. Configs control widths, block counts, datasets and staged
schedules. Prepared datasets, resumable training, single-node DDP, real-bitstream
validation and model export use the same CLI. See [training instructions](docs/TRAINING.md)
and the short [CUDA smoke recipe](configs/train/smoke.yaml).

### Deterministic INT16 execution

Set `DCVC_USE_INT16=1` to enable the integer runtime:

- Features use signed INT16 with scale `512`.
- Convolution weights use signed INT16 with scale `8192`.
- Convolution accumulation uses INT32 with explicit wrap, rounding, and clamp behavior.
- Sigmoid-derived nonlinear operations and entropy scale-index mapping use precomputed lookup tables.
- Quantized weights, biases, feature parameters, entropy tables, and lookup tables can be cached beside the original checkpoints as `.int16prep.pt` files.
- Custom CUDA kernels cover convolution, bias, residual arithmetic, reciprocal scaling, LUT application, entropy preparation, depthwise/activation fusion, and pixel-shuffle paths.
- If the extension is missing or `DCVC_USE_INT16` is disabled, the original floating-point path remains available.

See [docs/INT16_DESIGN.md](docs/INT16_DESIGN.md) for the arithmetic and compatibility details.

### Bit-exact kernel and pipeline optimization

The CUDA extension adds tiled implicit-GEMM fast paths for common dense INT16 convolutions:

- group-1, stride-1, unpadded `1x1`
- group-1, stride-1, pad-1 `3x3`

Unsupported shapes continue through the generic kernel. The optimized kernels preserve traversal, 32-bit wrap, rounding, and clipping behavior. Redundant device-wide synchronization after each encoded frame was removed; normal CUDA stream ordering and the blocking entropy boundary preserve dependencies.

On the development Jetson Orin fixture, the steady codec loop changed as follows:

| Measurement | Reference INT16 | Optimized | Change |
| --- | ---: | ---: | ---: |
| Encode loop | 21.9 fps | 38.6 fps | 1.76x |
| Decode loop | 21.9 fps | 37.6 fps | 1.72x |

These are small-resolution, device-specific engineering measurements—not reproductions of the paper's 1080p A100 results. The reference and optimized paths produced byte-identical bitstreams and decoded YUV in the local regression.

The CPU entropy encoder also uses exact reciprocal division, emits escaped
symbols without temporary heap allocations, and avoids waking the inactive
second coder. Rebuild the CPU extension to enable these changes; no new CLI
option or prepared-model conversion is needed. See the
[CPU measurements and compatibility checks](CPU_OPTIMIZATION_20260908.md).

### Unified media and archive workflow

`main.py` provides five commands:

```text
encode   FFmpeg input -> DCVC bitstream plus optional Opus audio and intra-coded thumbnails
stream   yt-dlp remote input -> DCVC bitstream plus optional Opus, without a source file
decode   DCVC bitstream -> decoded YUV/video workflow
view     Interactive bitstream viewer
cleanup  Revalidate a completed channel and remove exact inventoried sources
```

The encoder supports per-channel queues, resumable progress logs, atomic output commits, concurrent audio encoding, optional thumbnail intra coding, and a terminal dashboard with worker, channel, approval, and event views. Its overall ETA uses the average wall time of successfully completed videos divided across the active worker count; it appears after the first video finishes and intentionally remains a simple estimate.

Local folder encoding can also be shared across machines with `--shared-work`. Peers using the same shared output directory atomically claim different videos, publish only while they still own a renewable lease, expose owner/frame/FPS status to one another, and reject incompatible model or encode settings. Shared mode retains originals; run one normal encode/cleanup pass after every peer exits. See [cooperative multi-machine encoding](docs/SHARED_WORK.md).

The decoder accepts either a directory through `--input_folder` or one specific bitstream through `--input_file`; these options are mutually exclusive. Video `.bin` files decode to YUV, while standalone thumbnail `.dcvci` files decode to PNG. Use `--input_kind images` or `--input_kind videos` to select one kind when a channel contains both.

`main.py view` opens a `.bin` video, one `.dcvci` image, or a live gallery for a folder of `.dcvci` images. The viewer decodes frames directly to memory and does not create YUV or PNG output. Folder galleries keep the image model loaded and support the buttons, slider, Left/Right arrows, and Home/End keys.

DCVC bitstreams do not carry a marker identifying whether float or INT16 arithmetic produced them. Both decoder commands therefore accept `--runtime int16`, which verifies CUDA and the native INT16 extension before loading a model and refuses to fall back to float. `--runtime auto` preserves the environment-based behavior and prints the runtime it selected.

INT16 depth blocks fuse dense 1x1 convolution and residual addition when the native
extension supports it. Rebuild the extension after updating to enable this path;
older extensions continue to use separate operations. Both paths preserve the
convolution's clipping before the residual addition.

Optional `DCVC_INT16_AUTOTUNE=1` measures 16-, 32-, and 64-column tiles for these
fused convolutions on each CUDA device and input shape. Choices are cached for the
current process. This adds timing and synchronization on first use of each shape;
it does not change codec arithmetic or prepared models. The default uses 32
columns without tuning. During CUDA Graph capture, an unseen shape uses the
default instead of running timing operations. Set the variable alongside
`DCVC_USE_INT16=1` when launching the encoder or decoder.

See [residual fusion measurements](OPTIMIZATION_20260907.md) for validation,
tile benchmarks, and CUDA Graph evaluation.

Optional `DCVC_INT16_CUDA_GRAPH=1` replays P-frame feature extraction, the main
encoder, and the hyper-encoder together as a CUDA Graph when encoding with
`DCVC_USE_INT16=1`. It works with both `encode` and
`stream`, including multiple GPU workers. Each worker keeps one graph for the
current input shapes and CUDA stream, copies the current frame, reference features,
and QP scales into it, and rebuilds it when the shape changes. First use adds warmup
and capture overhead; retained graph buffers also use GPU memory. The default is
off. Decoding and the bitstream format are unchanged, so decoders need no new flag.
Use `DCVC_INT16_CUDA_GRAPH=feature` to compare with the earlier feature-only graph,
or `DCVC_INT16_CUDA_GRAPH=0` to disable capture.

See [CUDA Graph measurements and checks](OPTIMIZATION_20260911.md) before enabling
it on another GPU. Add `DCVC_INT16_CUDA_GRAPH=1` alongside your existing environment
variables; no native extension rebuild is required for this change.

Optional `DCVC_INT16_FAST_INPUT=1` speeds up frame preparation for INT16 `encode`
and `stream`. It replaces CPU chroma interpolation and per-frame floating-point
normalization with exact lookup-table conversion, nearest-neighbor chroma
replication, and reusable pinned upload buffers. It preserves the existing
float32-to-float16-to-INT16 rounding and edge padding. Add it alongside
`DCVC_USE_INT16=1`; it works with or without CUDA Graphs and is off by default.
The measured gain with the encoder graph enabled was **5.5%** on the longer Jetson
pipeline benchmark. See [input, transfer, and graph experiments](PIPELINE_OPTIMIZATION_20260911.md)
for measurements, exact-output checks, and the alternatives that were rejected.

### Multi-GPU execution

Encoding and decoding use file-level data parallelism. Each worker owns a complete model instance, selects one logical CUDA device before loading that model, and processes a whole video on that device. Videos never migrate between GPUs, and the codec arithmetic and bitstream syntax are unchanged.

When `--procs` (encode) or `--worker` (decode) is omitted, the CLI starts one worker per visible GPU. Use `--cuda_idx` to select a subset; the indices are the logical PyTorch indices after `CUDA_VISIBLE_DEVICES` is applied. For example, this uses five GPUs:

```bash
DCVC_USE_INT16=1 python main.py encode \
  --base_root /data/incoming \
  --output_root /data/encoded \
  --channel_ids CHANNEL_A CHANNEL_B \
  --cuda_idx 0 1 2 3 4
```

An explicit worker count is assigned round-robin across the selected devices. More than one worker per GPU is allowed but is not recommended unless model-memory headroom has been measured. If fewer videos remain than planned workers, only the required workers are launched. The selected worker-to-GPU mapping is printed and shown in the dashboard.

Prepared INT16 checkpoint caches are published atomically, so simultaneous first-launch workers cannot observe a partially written cache. Worker cleanup is also limited to temporary files registered by that worker.

### Direct yt-dlp streaming

The `stream` command accepts individual videos, playlists, YouTube channel IDs, Twitch channel names, and other URLs supported by yt-dlp. Listing and metadata resolution use `--skip-download`; for encoding, yt-dlp writes one selected media stream to standard output and FFmpeg reads it from a pipe. Decoded YUV travels through the bounded in-memory frame queue to the neural encoder. No original source container is written.

The archive outputs are still stored normally: `.bin`, optional `.opus`, `.info.json`, and an optional thumbnail. Interrupted or failed final outputs remain protected by the same atomic-write behavior as local encoding. Completed remote items are recognized from their final artifacts and skipped on a later run.

Local and streamed encoding report each video's dimensions as `source width, source height -> output width, output height`. When the computed output dimensions already equal the source dimensions, FFmpeg omits the scale filter.

yt-dlp remains an external runtime tool and is not copied into the Conda environment. The command searches `PATH` and the directory containing `CONDA_EXE`, or it can be selected explicitly with `--yt-dlp`. Current YouTube extraction defaults to yt-dlp's `web_safari` player client and permits its recommended EJS challenge component; both the rationale and override are documented in [docs/STREAMING.md](docs/STREAMING.md).

Channel cleanup is deliberately fail-closed. Before a source can be deleted, the lifecycle verifies its original identity, the latest encode status, DCVC bitstream structure and frame count, Opus stream validity, metadata JSON, retained or intra-coded thumbnails, and safe path boundaries. It writes an archive manifest and append-only audit record before unlinking exact paths. Shell globs and recursive deletion are never used.

See [docs/MANAGED_ARCHIVE.md](docs/MANAGED_ARCHIVE.md) for operating details.

## Relationship to the CVPR 2025 paper

The [DCVC-RT paper](https://openaccess.thecvf.com/content/CVPR2025/html/Jia_Towards_Practical_Real-Time_Neural_Video_Compression_CVPR_2025_paper.html) introduces the neural architecture, implicit temporal modeling, low-resolution latent representation, module-bank rate control, and training-free 16-bit model integerization. This fork retains those upstream models and checkpoints.

The existing pretrained checkpoints remain the inference defaults; this fork does not claim a new rate-distortion result. Optional training now supports configurable architectures and INT16-aware fine-tuning. Its integer scales follow the paper's integerization design, while the training implementation, PyTorch conversion/cache code, CUDA arithmetic, extension loader, fused operations, exact-output tests, FFmpeg/Opus frontend, TUI, and archival lifecycle extend what the paper specifies.

Important compatibility distinction:

- The bitstream syntax is not intentionally changed.
- The optimized kernels are byte-identical to this fork's reference INT16 implementation in the tested environment.
- Deterministic predictive decoding should use the INT16 runtime on both sides. The upstream floating-point runtime is not claimed to reproduce every integer intermediate state across devices.
- A second physical GPU architecture was not available for the original regression, so cross-device execution remains a design property that should be verified on each supported architecture before archival deployment.

## Installation

Create and activate a Conda environment containing CUDA-enabled PyTorch and the Python dependencies from `requirements.txt`. Then build both native extensions:

```bash
conda activate <environment>
./build_native_extensions.sh
```

The script validates that Python belongs to the active Conda environment, detects NVCC and every visible CUDA architecture, cleanly builds the CPU entropy coder and CUDA/INT16 extension, installs them into that environment, refreshes the source-tree CUDA binary, and verifies all required integer symbols. It builds native code for the current host, so run it again after copying the repository to an x86-64 machine; do not copy the Jetson/AArch64 `.so` files.

By default `MAX_JOBS=2` reduces compilation memory pressure. If no GPU is visible while compiling, specify the target architecture explicitly, for example:

```bash
TORCH_CUDA_ARCH_LIST="8.7+PTX" ./build_native_extensions.sh
```

Checkpoints are intentionally not stored in Git. Download the official DCVC-RT checkpoints as described in [UPSTREAM_README.md](UPSTREAM_README.md) and place or link them under `checkpoints/`. Prepared `.int16prep.pt` caches are generated locally and should also remain outside Git.

## Encoding example

For automatic rate/quality tuning on one source, see the
[INT16 Pareto search](docs/PARETO_SEARCH.md): 800 balanced initial samples followed
by genetic search, with atomic JSON progress and no retained test bitstreams.

```bash
DCVC_USE_INT16=1 python main.py encode \
  --base_root /data/incoming \
  --output_root /data/encoded \
  --channel_ids CHANNEL_A CHANNEL_B \
  --model_path_i checkpoints/cvpr2025_image.pth.tar \
  --model_path_p checkpoints/cvpr2025_video.pth.tar \
  --resolution 96 --fps 24 --qp_i 35 --qp_p 14 \
  --audio opus --opus_bitrate 6k --opus_channels mono \
  --cuda_idx 0 1 2 3 4 --ui auto
```

With no cleanup option, completed channels await approval while other channels continue encoding. Add `--auto-delete` for validation-gated unattended cleanup, `--cleanup-dry-run` to exercise every check without unlinking, or `--keep-originals` to disable cleanup.

To compress every `.webp`, `.jpg`, `.jpeg`, and `.png` in each selected channel with the image model, add:

```bash
--thumbnail_codec dcvc-intra --thumbnail_qp 45
```

Each image keeps its exact dimensions and is stored as `<original-name>_qI45.dcvci`, containing one SPS and one I-frame. The phase has its own resumable `.thumbnail-progress.jsonl`, commits outputs atomically, and round-trip decodes each result before it becomes eligible for cleanup. It loads only the image model and defaults to one thumbnail worker per selected GPU; use `--thumbnail_workers` to override that count. In the tested INT16 checkpoint, QP 45 is a practical default and quality degraded above QP 47, so higher values print a warning.

With `DCVC_INT16_AUTOTUNE=1`, each thumbnail worker benchmarks new kernel shapes
only while encoding and verifying its first image. Later images reuse cached
tiles; unseen shapes use the default 32-column tile without timing trials. This
avoids repeated tuning when thumbnails have different dimensions. Autotuning
tests individual convolution kernels, not whole images or CUDA Graphs. A failed
encoding attempt also ends that worker's tuning window; failures before image
loading completes do not. Every image still receives its round-trip decode check,
and video autotuning keeps its normal per-shape behavior.

When this mode is active, validated original thumbnail files join the same cleanup approval as the source videos. `--keep-originals` retains both. Shared workers defer thumbnails; after every `--shared-work` peer exits, run one ordinary encode pass with the thumbnail options to encode them, build the final inventory, and offer cleanup.

Decode one archived thumbnail to PNG:

```bash
DCVC_INT16_AUTOTUNE=1 python main.py decode \
  --input_file /data/encoded/CHANNEL/video.webp_qI45.dcvci \
  --output_folder /data/decoded \
  --cuda true --runtime int16
```

Decode every archived image in a channel while ignoring its `.bin` videos:

```bash
DCVC_INT16_AUTOTUNE=1 python main.py decode \
  --input_folder /data/encoded/CHANNEL \
  --input_kind images \
  --output_folder /data/decoded \
  --cuda true --runtime int16
```

Browse those images live without writing decoded files:

```bash
DCVC_INT16_AUTOTUNE=1 python main.py view \
  /data/encoded/CHANNEL \
  --cuda true --runtime int16
```

The same explicit runtime fixes predictive live playback for an INT16 video and reports `Decoder runtime: CUDA INT16` before opening the window:

```bash
DCVC_INT16_AUTOTUNE=1 python main.py view \
  /data/encoded/CHANNEL/video.bin \
  --cuda true --runtime int16
```

On a single-GPU Jetson, the default remains one worker. On a multi-GPU host, the default is one worker per visible GPU. Use `--procs` to override encoding concurrency or `--worker` to override decoding concurrency.

## Streaming example

```bash
DCVC_USE_INT16=1 python main.py stream \
  --youtube_channels UCxxxxxxxxxxxxxxxxxxxxxx ./UCyyyyyyyyyyyyyyyyyyyyyy.txt \
  --output_root /data/encoded \
  --model_path_i checkpoints/cvpr2025_image.pth.tar \
  --model_path_p checkpoints/cvpr2025_video.pth.tar \
  --resolution 96 --fps 24 --qp_i 35 --qp_p 14 \
  --audio opus --opus_bitrate 6k --opus_channels mono \
  --cuda_idx 0 1 2 3 4 --ui auto
```

For an initial end-to-end check, add `--max-videos 1`. Private or age-restricted sources can use `--cookies /path/to/cookies.txt`. The supplied file remains unchanged: every yt-dlp invocation receives its own temporary RAM copy with mode `660`, and that copy is removed afterward. A read-only backup can therefore be passed directly. See [direct remote streaming](docs/STREAMING.md) for authenticated YouTube client selection.

## Tests

The unit suite creates its own synthetic media and does not require redistributed video:

```bash
python -m unittest discover -s tests -v
```

GPU microbenchmark:

```bash
DCVC_USE_INT16=1 python benchmarks/benchmark_int16_conv.py
```

## Contributors and provenance

- **Original research and upstream implementation:** Zhaoyang Jia, Bin Li, Jiahao Li, Wenxuan Xie, Linfeng Qi, Houqiang Li, Yan Lu, Microsoft Research, and the broader DCVC contributors.
- **Fork owner and maintainer:** [Salatiel Jordão (@QLaHPD)](https://github.com/QLaHPD).
- **Primary implementation contributor for the INT16 optimization and managed-runtime work:** **OpenAI Codex**, operating under the owner's direction and review.

OpenAI's public Codex material describes Codex as a coding agent used to build, refactor, test, and maintain code; it does not define a special GitHub co-author identity. Accordingly, this repository credits Codex in documentation without inventing an account or email. The human repository owner remains responsible for review, publication, and maintenance.

This fork is not affiliated with or endorsed by Microsoft or OpenAI. Upstream history and the repository's [MIT License](../../LICENSE.txt) are preserved.
