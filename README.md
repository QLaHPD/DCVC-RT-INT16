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

### YouTube streaming

`stream` (also `encode --source_urls`) feeds yt-dlp output directly to FFmpeg and the UF encoder, without keeping downloaded source videos. Completed, decoded-and-validated files (`VIDEO.bin`, `VIDEO.opus`, `VIDEO.uf.json`) are stored directly under the resolved YouTube channel ID, without a per-video directory. Existing archives retain the normal compatibility and integrity checks. Each video uses a fresh worker; this streaming mode currently supports `--procs 1` and full videos only.

```bash
./scripts/uf-python main.py stream --source_urls https://www.youtube.com/@FattoincasadaBenedettaOfficial --output_root /mnt/to_storage/DATA/YOUTUBE --runtime int16 --model_structure htl --resolution 144 --procs 1 --audio opus --opus_channels mono --opus_bitrate 6k --qp_i 10 --qp_p 14 --reset_interval 2 --intra_period -1 --skip_thres 0.2 --ytdlp_bin /path/to/yt-dlp
```

Omit `--fps` to preserve source frame rate. Video input uses the highest available resolution and is resized with preserved aspect ratio. Audio favors tracks identified as original; ambiguous multiple-language selections fail rather than choosing a dub silently. Optional `--cookies /path/to/cookies.txt` uses a temporary mode-660 copy for each yt-dlp invocation, leaving the supplied file unchanged. Download failures never publish an incomplete archive.

The example configuration was the fastest measured UF HT-L INT16 candidate above 30 dB on the 144p Bunny search. Fixed encoder settings do not guarantee that PSNR on other videos. Streaming sources have a URL identity rather than a local source-file checksum, and local-original cleanup does not apply to them.

### Managed interface and original cleanup

Use `--ui plain` for timestamped logs or `--ui tui` for frame/FPS progress and channel cleanup controls (`auto` selects based on terminal availability). After encoding, TUI stays open for review; select a channel, press **d**, then **y** to approve verified original deletion. **a** requests cleanup for all eligible listed channels; **q** exits. `--auto-delete` performs verified cleanup automatically; `--keep-originals` retains inputs and exits. Streamed inputs have no stored original video to delete.

Existing outputs can be reviewed separately:

```bash
./scripts/uf-python main.py manage --output_root /mnt/to_storage/DATA/YOUTUBE --channel_ids CHANNEL_ID --ui tui
```

[Storage compatibility, deletion safeguards, and plain-mode commands](docs/UF_ARCHIVE.md).
