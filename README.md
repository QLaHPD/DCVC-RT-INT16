# DCVC-UF INT16 development

Local development branch for integer inference and portable decoding in
[Microsoft's DCVC-UF](https://github.com/microsoft/DCVC).

**Current status:** the archive CLI supports FP16 (default) and a prepared INT16
runtime selected with `--runtime int16`. Both have completed a full 720p Bunny
encode/decode validation on Jetson. Integer arithmetic also has an independent
CPU reference; additional physical GPU models still need validation.
See [integer runtime and portability](docs/UF_INT16.md),
[archive commands](docs/UF_ARCHIVE.md), and [FP16 baseline](docs/UF_BUNNY_VALIDATION.md).
The [Pareto search](docs/UF_PARETO.md) finds nondominated size/PSNR settings
without retaining its temporary bitstreams.

## Local workspace

- Checkout: `/mnt/to_storage/TOOLS/DCVC-UF-INT16`
- Branch: `uf-int16-managed`
- Dedicated environment: `dcvc-uf-int16`
- Models: **[checkpoints/](checkpoints/README.md)**
- Setup instructions and compatibility status: **[docs/UF_LOCAL_SETUP.md](docs/UF_LOCAL_SETUP.md)**

```bash
./scripts/uf-python scripts/check_uf_setup.py
./scripts/uf-python test_video.py --help
```

The launcher selects the dedicated local interpreter and cache directories. It
removes inherited RT INT16 flags and Python import-path overrides for its child
process. No checkpoints are downloaded automatically.

## Source layout

| Location | Purpose |
|---|---|
| `src/models/image_model.py` | UF intra-frame model |
| `src/models/video_model_ht.py` | UF HT-S / HT-L models |
| `src/models/video_model_ld.py` | UF low-delay model |
| `src/int16/` | Prepared integer models, CPU reference and integer CUDA kernels |
| `main.py`, `src/archive/` | Archive CLI, runtime selection and storage workflow |
| `test_video.py` | Upstream UF evaluation and bitstream coding interface |
| `train_image.py`, `train_video.py`, `training.md` | Upstream UF float training |
| `DCVC-family/DCVC-RT/` | Inherited RT implementation, maintained separately |

Development starts from RT branch commit `468bd15`. This is a separate Git
worktree with its own branch and working files; it shares Git history with the RT
checkout. UF development is published on `uf-int16-managed`; RT development
remains on `int16-managed`.

Keep model files, native binaries, caches, datasets and outputs out of Git. The
existing RT runtime and its prepared INT16 files remain in their original location.

### Choose settings by benchmark PSNR

Add `--target-psnr 30` to `encode` or `stream` to select the fastest measured
configuration with PSNR ≥30 dB; smaller bitstream size breaks equal-time ties.
This fills INT16, HT-L, 144p and the five codec parameters automatically.
Explicit codec parameters constrain the candidate search; omit the target to
keep fully manual settings. The target is a **benchmark reference**, not a
quality guarantee for new content. See the [published 2,000-trial benchmark](benchmarks/README.md).

```bash
./scripts/uf-python main.py encode --input_file video.mp4 --output_root output --target-psnr 30
```

### YouTube streaming

`stream` (also `encode --source_urls`) feeds yt-dlp output directly to FFmpeg and the UF encoder, without keeping downloaded source videos. Completed, hash-recorded files (`VIDEO_ID_UNIX_TIMESTAMP_WIDTHxHEIGHT_qIQI_qPQP.bin`, matching `.opus`, and `.uf.json`) are stored directly under the resolved YouTube channel ID, without a per-video directory. Existing archives retain the normal compatibility and integrity checks. Each video uses a fresh worker; `--procs` and `--cuda_idx` support multiple GPUs. Streaming processes complete videos. Input buffering starts at eight frames and calibrates once after five seconds to two seconds of measured encoding throughput. The next video download starts near the current download’s end to overlap yt-dlp startup; see [streaming buffering details](docs/UF_ARCHIVE.md#encode-and-stream).

```bash
./scripts/uf-python main.py stream --source_urls https://www.youtube.com/@FattoincasadaBenedettaOfficial --output_root /mnt/to_storage/DATA/YOUTUBE --runtime int16 --model_structure htl --resolution 144 --procs 1 --audio opus --opus_channels mono --opus_bitrate 6k --qp_i 10 --qp_p 14 --reset_interval 2 --intra_period -1 --skip_thres 0.2 --ytdlp_bin /path/to/yt-dlp
```

Omit `--fps` to preserve source frame rate. Video download uses RT’s 480p cap (`--source-max-height 0` disables it), then resizes with preserved aspect ratio. Audio favors tracks identified as original; ambiguous multiple-language selections fail rather than choosing a dub silently. Optional `--cookies /path/to/cookies.txt` uses a temporary mode-660 copy for each yt-dlp invocation, leaving the supplied file unchanged. Download failures never publish an incomplete archive.

The example configuration was the fastest measured UF HT-L INT16 candidate above 30 dB on the 144p Bunny search. Fixed encoder settings do not guarantee that PSNR on other videos. Streaming sources have a URL identity rather than a local source-file checksum, and local-original cleanup does not apply to them.

### Managed interface and original cleanup

Use `--ui plain` for timestamped logs or `--ui tui` for frame/FPS progress and channel cleanup controls (`auto` selects based on terminal availability). After encoding, TUI stays open for review; select a channel, press **d**, then **y** to approve verified original deletion. The RT views use **w/c/a/e**; **k** keeps originals, **r** retries validation, and **x** exits after active work. `--auto-delete` performs verified cleanup automatically; `--keep-originals` retains inputs and exits. Streamed inputs have no stored original video to delete.

Existing outputs can be reviewed separately:

```bash
./scripts/uf-python main.py manage --output_root /mnt/to_storage/DATA/YOUTUBE --channel_ids CHANNEL_ID --ui tui
```

[Storage compatibility, deletion safeguards, and plain-mode commands](docs/UF_ARCHIVE.md).

### Thumbnails and seekable playback

Add `--thumbnail_codec dcvc-intra --thumbnail_qp 45` to the same encode/stream
command. Images become `original.jpg_qI45.dcvci`, using UF's intra network and
recorded runtime/model identities. Images-only input does not need a video model.

```bash
./scripts/uf-python main.py decode --input_folder CHANNEL_FOLDER --output_folder /media/ramdisk/decoded --input_kind all
./scripts/uf-python main.py view CHANNEL_FOLDER/VIDEO_256x144_qI10_qP14.bin
./scripts/uf-python main.py view CHANNEL_FOLDER
```

The RT viewer controls now drive UF decoding: video seeking starts at the nearest
I-frame, and folders open an intra-image gallery. Both decode in memory without
saving media. `--runtime auto` uses the recorded arithmetic, including INT16.
[Commands, archive compatibility and validation limits](docs/UF_ARCHIVE.md).

Encoding publishes after encoding and artifact hashing, without an automatic neural decode pass. `verify` remains available explicitly; existing decoded-pixel validation records are still honored.
