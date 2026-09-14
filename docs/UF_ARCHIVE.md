# Managed UF archives

UF uses the RT managed dashboard and playback widgets, with UF model loading,
INT16 arithmetic, temporal packets and CUDA kernels behind them. RT bitstreams
and UF bitstreams require their respective networks. This does not convert an
existing RT archive to UF.

## Encode and stream

```bash
./scripts/uf-python main.py encode --base_root /mnt/to_storage/DATA/YOUTUBE --output_root /mnt/to_storage/DATA/YOUTUBE --channel_ids CHANNEL_ID --runtime int16 --model_structure htl --resolution 144 --fps 24 --procs 1 --cuda true --ui tui --opus_channels mono --opus_bitrate 6k --qp_i 10 --qp_p 14 --reset_interval 2 --skip_thres 0.2 --thumbnail_codec dcvc-intra --thumbnail_qp 45
```

Local input accepts `--input_file` or `--base_root`, optionally limited by
`--channel_ids`. Add `--thumbnail_codec dcvc-intra` to encode source images as
well as videos; `keep` retains images unchanged. Images-only folders and a single
image passed through `--input_file` work without a video checkpoint.

Streaming uses the same encoder and multi-worker scheduler:

```bash
./scripts/uf-python main.py stream --source_urls https://www.youtube.com/@FattoincasadaBenedettaOfficial --output_root /mnt/to_storage/DATA/YOUTUBE --runtime int16 --model_structure htl --resolution 144 --procs 1 --cuda_idx 0 --ui plain --opus_channels mono --opus_bitrate 6k --qp_i 10 --qp_p 14 --reset_interval 2 --skip_thres 0.2 --cookies /path/to/cookies.txt --yt-dlp /path/to/yt-dlp --thumbnail_codec dcvc-intra
```

`encode --source_urls` is equivalent. `--youtube_channels` accepts channel IDs
or text files containing one video ID per line. `--twitch_channels` selects
Twitch archive listings. Streaming handles full, completed videos; current live
and upcoming broadcasts are skipped. `--max-videos` limits discovery.

The resolved channel ID determines the output folder. Source video selection
uses RT's 480p download cap (`--source-max-height 0` allows any resolution).
The requested encoding resolution is applied afterward. Original audio tracks
are preferred; ambiguous multilingual tracks fail rather than silently choosing
a dub. Each yt-dlp invocation uses a temporary mode-660 cookie copy, leaving the
supplied cookies unchanged. Thumbnails are retained by default; `--no-thumbnail`
disables their download. A thumbnail download failure does not discard a valid
video archive.

UF keeps its established codec/input defaults: FP16, HT-S, original dimensions
and FPS, one process, QP I=36/P=30, and mono Opus at 6 kbps. Pass the desired
settings explicitly when moving an RT command. `--resolution 144` preserves
aspect ratio (1280×720 → 256×144); matching dimensions skip scaling. Odd video
dimensions are made even for YUV420. `--fps 8` samples eight frames per second.
HT-L/HT-S code eight-frame P packets, so positive `--intra_period` values must be
1 or multiples of eight; LD accepts any positive interval.

`--opus_frame_ms` defaults to 60, with `--opus_complexity 10 --opus_vbr on`.
An older UF run used FFmpeg's default 20 ms; specify `--opus_frame_ms 20` when
resuming that pipeline. Existing valid sibling `.opus` files are reused, including
when re-encoding only video. `--audio none` prevents source deletion when the
source contains audio.

`--input_threads` sets the FFmpeg thread budget. INT16 defaults to up to four
threads per worker and `--prefetch_frames 8` (alias `--ffmpeg_prefetch`); queued
frames are bounded to 8 MiB or one frame. `--ff_hwaccel none|auto|jetson` selects
the decoder. The Jetson option reuses RT's GStreamer NVDEC bridge for supported
local H.264 inputs, with a complete software retry on hardware decode failure.

## Names, resume and shared work

For a local `clip.mkv`, outputs are:

```text
clip_256x144_qI10_qP14.bin
clip.opus
clip.uf.json
clip.info.json                 (when provided)
clip.jpg_qI45.dcvci             (when thumbnail encoding is enabled)
```

A newly streamed video's basename is `VIDEO_ID_UPLOAD_DATE`, just as in RT.
For duplicate local basenames, MKV is preferred, followed by MP4, WebM, MOV and
AVI. Alternate containers remain untouched. Relative input folders are retained.

Keep `.uf.json` with its video bitstream: it contains runtime, prepared-model
identities, frame counts, source identity, artifact hashes and validation results.
Older flat names and `<filename>.uf/` bundles remain readable and resumable.
An already-running process keeps its original output layout.

```bash
./scripts/uf-python main.py encode --base_root /local/mount/DATA/YOUTUBE --output_root /local/mount/DATA/YOUTUBE --channel_ids CHANNEL_ID --runtime int16 --model_structure htl --resolution 144 --fps 24 --procs 5 --cuda_idx 0 1 2 3 4 --shared-work --shared-instance desktop --ui plain --opus_channels mono --opus_bitrate 6k --qp_i 10 --qp_p 14 --reset_interval 2 --skip_thres 0.2
```

Use the same encoding settings on every machine and different instance names.
Mount paths and worker/GPU counts may differ. UF always uses cooperative claims;
`--shared-lease-seconds` defaults to 900. Video and image claims use separate
namespaces. Payload publication never overwrites an existing different file.
Hidden staging plus a manifest-last commit makes interrupted video publication
recoverable. Use separate output directories for different video configurations.

## Dashboard and cleanup

The actual RT dashboard provides Workers (**w**), Channels (**c**), Approvals
(**a**) and Events (**e**), with Tab/arrow navigation. In Approvals, **d** requests
exact-path deletion and **y** confirms; **k** retains originals, **r** retries
validation, and **x** exits approvals after active work finishes. TUI stays open
when all videos were already encoded. `--ui plain` logs progress without waiting
for approval; `auto` chooses based on terminal availability.

`--auto-delete` verifies and removes eligible originals automatically.
`--keep-originals` exits with originals retained; `--cleanup-dry-run` previews.
Failed/busy work, unprocessed videos, unmatched metadata and missing `info.json`
block channel deletion. `--allow-missing-metadata` permits the last condition;
it does not bypass integrity checks. Cleanup only removes the exact original
video/image represented by a verified archive, never its directory, alternate
containers or unrelated sidecars. Other instances' deletions appear on refresh.

Review an existing channel without encoding:

```bash
./scripts/uf-python main.py manage --output_root /local/mount/DATA/YOUTUBE --base_root /local/mount/DATA/YOUTUBE --channel_ids CHANNEL_ID --ui tui
```

`--base_root` remaps recorded relative source paths when another machine created
the archive. Streamed videos have no stored original video to delete.

For one original, `cleanup --input_file ARCHIVE` previews;
`cleanup --input_file ARCHIVE --yes` verifies and deletes it. Both `.bin` and
`.dcvci` inputs work. Partial video archives cannot authorize source deletion.

## Decode, seek and view images

```bash
./scripts/uf-python main.py decode --input_file channel/clip_256x144_qI10_qP14.bin --output_file /media/ramdisk/clip.mkv
./scripts/uf-python main.py decode --input_folder channel --output_folder /media/ramdisk/decoded --input_kind all --worker 1 --cuda_idx 0
./scripts/uf-python main.py decode --input_file channel/clip.jpg_qI45.dcvci --output_file /media/ramdisk/clip.png
./scripts/uf-python main.py view channel/clip_256x144_qI10_qP14.bin
./scripts/uf-python main.py view channel/clip.jpg_qI45.dcvci
./scripts/uf-python main.py view channel --recursive
```

Folder decoding selects `all`, `videos` or `images`, writing raw YUV420 `.yuv`
video and `.png` images, as in RT. `--worker`/`-w` and `--cuda_idx 0 1 ...` allow
multiple decoding workers. Explicit `.mkv` output uses FFV1 and copies archived
audio. Output files are published only after verification and never replaced.

The shared RT Tk viewer provides Play/Pause, previous/next frame, a seek slider,
`--start_frame`, `--fps`, and display size limits. It decodes in memory without
writing reconstructed media. Seeking starts at the nearest preceding I-frame;
with `--intra_period -1`, a backward jump may require decoding from the start.
UF's padded final temporal packet is trimmed to the recorded frame count.
Like RT's viewer, playback is video-only, without synchronized audio. Folder
image viewing provides a gallery and reuses the intra model between images.

UF `.dcvci` files are self-contained: they embed their pipeline, original image
size, payload hash and decoded-pixel validation. They use the UF intra network,
not RT's intra network. The encoder reuses one intra model for the image batch
and does not load the temporal model for image-only input. Odd image dimensions
are padded internally and cropped back on PNG export/display.

`--runtime auto` reads the archive's recorded arithmetic. An explicitly wrong
runtime or prepared-model identity fails; the viewer does not guess float for
an INT16 file. `verify --input_file ARCHIVE` checks reconstruction without saving.
INT16 also supports `--device cpu` as a reference backend. FP16 requires CUDA;
cross-GPU FP16 bit-exact reproduction is not guaranteed.

## Validation of this integration

Regression tests cover storage collisions, interrupted publication, leases,
source remapping, Opus reuse, cookie isolation, source selection, intra files,
image model reuse and temporal seeking. A real odd-sized INT16 image passed CPU
encode/decode, CPU-to-GPU verification and the CLI image-only path with a missing
video checkpoint. Tk/Pillow import successfully. The concurrent FP16 GPU image
smoke test exhausted Jetson memory; it remains to be rerun with sufficient free
GPU memory. The live encoder was left running. No interactive display or five-GPU
hardware test was performed during this integration.
