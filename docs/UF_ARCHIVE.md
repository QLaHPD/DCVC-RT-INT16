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

New channels use the established codec/input defaults: FP16, HT-S, original dimensions
and FPS, one process, QP I=36/P=30, and mono Opus at 6 kbps. Pass the desired
settings explicitly when moving an RT command. `--resolution 144` preserves
aspect ratio (1280×720 → 256×144); matching dimensions skip scaling. Odd video
dimensions are made even for YUV420. `--fps 8` samples eight frames per second.
HT-L/HT-S code eight-frame P packets, so positive `--intra_period` values must be
1 or multiples of eight; LD accepts any positive interval.

Local INT16 encoding keeps up to `--procs` worker processes alive. Each worker
loads its checked image/video model pair only when it first needs to encode,
then reuses those weights across videos. Video references are cleared between
archives, and resolution-dependent mask caches retain one shape. Failed workers
are replaced before taking another video. CPU threads hash/probe at most one
extra local input per worker while the GPU processes current videos. A changed
input is prepared again; the final source checksum is still checked before
publication. FP16 and streaming retain their existing process lifecycle.
On a resumed run, existing video archives undergo all resume
integrity checks first. Their results enter the progress counter before the
remaining videos start encoding. Bounded CPU threads check artifact hashes and
the original checksum when the original still exists, without neural decoding or
encoder model initialization. Worker reuse does not change saved pipeline settings.

`--opus_frame_ms` defaults to 60, with `--opus_complexity 10 --opus_vbr on`.
An older UF run used FFmpeg's default 20 ms; saved profiles inherit that value
automatically. Existing valid sibling `.opus` files are reused, including
when re-encoding only video. `--audio none` prevents source deletion when the
source contains audio.

`--input_threads` sets the FFmpeg thread budget. INT16 defaults to up to four
threads per worker and `--prefetch_frames 8` (alias `--ffmpeg_prefetch`); queued
local-file queues are bounded to 8 MiB or one frame. `--ff_hwaccel none|auto|jetson` selects
the decoder. The Jetson option reuses RT's GStreamer NVDEC bridge for supported
local H.264 inputs, with a complete software retry on hardware decode failure.

Streaming instead primes an eight-frame decoded input queue. After the first
five seconds of encoding, it measures average encoding FPS once and changes the
queue capacity to `ceil(FPS * 2)` frames, shared by subsequent videos/workers
using the same profile. If the first video ends before five seconds, calibration waits for an
eligible later video. The queue fills ahead as download/decode speed permits;
it cannot compensate for a persistently slower source. Its YUV444 frame storage
uses approximately `frames * width * height * 3` bytes (about 15 MiB for 140
frames at 256x144), without the local-file queue's 8 MiB cap.

Next-video metadata resolves in the background. At 20 or fewer remaining video
fragments, one additional yt-dlp video stream starts while the current worker
finishes. Formats without fragment counts use 95% byte progress when available;
remaining encoded frames and download completion provide additional triggers.
The pending stream backpressures on a bounded OS pipe until a worker is free,
without saving a source video. Cookies still use private temporary copies, and
shared claims and atomic publication remain enforced by the encoding worker.
Restart an existing streaming run to activate this scheduler.

### Benchmark quality selection

`--target-psnr 30` works with local `encode`, `stream`, and `encode --source_urls`.
It uses the bundled [benchmark](../benchmarks/README.md), including all successful
trials, rather than only the size/quality Pareto frontier (which excludes some
fast configurations). Candidates must reach the requested PSNR; ordering is
encoding seconds, then bitstream bytes, then trial number for deterministic ties.
Time and size are competing objectives, so time has priority.

The selector fills `--runtime int16 --model_structure htl` and `--qp_i`,
`--qp_p`, `--reset_interval`, `--intra_period`, `--skip_thres`. Any explicit
`--resolution` is preserved; for a new channel, omitting it defaults to the
benchmark's 144p short edge. Saved profiles retain their resolution.
For example, `--target-psnr 30 --resolution 480` selects
parameters from the calibration table and encodes at a 480p short edge.
Explicit codec parameters, including aliases `--qi`, `--qp`, and
`--force_intra_period`, constrain selection and are retained. If no measured
candidate qualifies, the command fails before downloading or loading models.
Explicitly requesting another runtime or model structure also fails;
omit `--target-psnr` for unrestricted manual settings. Audio, process count,
source FPS and other options retain their usual behavior.

The selected trial, parameters and measured size/time/PSNR are printed in a
`target_psnr_selected` event. PSNR comes from the benchmark source, and speed
comes from the benchmark hardware. Different resolutions, content, frame rates,
prepared models or hardware can change both quality and performance. No per-video
PSNR measurement or search is added, and the selector does not change the output
format or pipeline identity beyond the actual selected codec settings.

### Skip completed channel entries before resolving videos

Streaming resolves the selected channel identities, loads each profile and checks
its committed archives before starting remaining video downloads or encoders.
Recorded artifact hashes, required audio and local source checksums where present
are checked. Each completed entry emits `already_encoded`, increments progress
once and contributes to the final `resumed` count.
`--max-videos` applies to remaining work. Nested channel tabs inherit channel IDs.
If the provider omits the channel ID in the listing, metadata must resolve it
before the same completion check can run.

`verify` additionally performs neural decoding. Invalid committed outputs are
reported as failures and retained; bare `.bin` names and hidden stages are not
completion evidence. Thumbnail downloads are not retried for completed videos;
existing local thumbnail originals can resume their saved UF image pass.
Concurrent peer completions receive the same artifact validation and count once.

### RAM staging and atomic output

Video bitstreams, encoded Opus and intra-image payloads are staged in Linux
`/dev/shm` (tmpfs) while encoding. There is no disk fallback: insufficient tmpfs
space fails the job and leaves final output unpublished. Budget RAM for the
completed compressed video/audio per worker in addition to models and frame
buffers. Downloaded source videos still pass through pipes.

Only after encoding and integrity checks finish are completed bytes copied into
a hidden stage on the destination filesystem, synced, and atomically linked to
final names without replacement. A same-filesystem copy is necessary because
RAM-to-disk rename/link cannot be atomic. The manifest commits the completed
video archive last. A crash during this final publication may leave a hidden
disk stage and pending journal for existing recovery logic. Older disk stages
are retained for recovery; changing the staging policy does not delete them.
Normal success/failure removes RAM staging; a forced kill can leave RAM files
until cleanup or reboot. tmpfs follows the operating system's swap policy.

Shared pipeline mismatches now report differing fields. Existing channel
pipelines that omit `opus_frame_ms` inherit the historical 20 ms default, even
though new channels default to 60 ms.

## Join and saved channel settings

```bash
# Help an existing local-file batch on a shared mount.
./scripts/uf-python main.py join --root /mounted/YOUTUBE \
  --channels UCzjtGnD7qqeaHW3nvDVrjQA https://www.youtube.com/@anotherchannel \
  --procs 2 --shared-instance desktop --ui plain

# A configured channel directory can be supplied directly.
./scripts/uf-python main.py join --root /mounted/YOUTUBE/CHANNEL_ID

# Update local originals later, using the same saved profile on one machine.
./scripts/uf-python main.py encode --root /mounted/YOUTUBE --channels CHANNEL_ID

# Fetch remote inputs using saved settings, without keeping downloaded videos.
./scripts/uf-python main.py stream --root /mounted/YOUTUBE \
  --channels https://www.youtube.com/@anotherchannel --cookies /local/cookies.txt
```

`--root` is the storage root; `--channels` accepts folder IDs or channel URLs and
deduplicates selections. Existing `--base_root`, `--output_root` and
`--channel_ids` remain available, including separate local input/output roots.
Channel URLs use local `info.json` aliases first, then metadata-only yt-dlp
resolution when needed. Ambiguous matches fail; select the exact folder ID.
The command does not establish SSH connections or mount filesystems. Supply your
existing filesystem mount, including a mount reached through Tailscale SSH.

Settings come from `.dcvc-uf-shared-work/pipeline.json`, or consistent video
manifests (`*.uf.json` or legacy `*.uf/manifest.json`) if no pipeline file exists.
Malformed profiles, conflicting manifests and incompatible manual overrides fail
before encoding begins. `join` requires a saved UF video profile. Ordinary
`encode` and `stream` keep their defaults for new channels. UF thumbnail settings
are recovered separately from `.dcvc-uf-image-work/pipeline.json` when present;
RT images and RT state are not UF profiles.

Inherited fields include runtime, variant, model identities, QPs, FPS, resolution,
temporal settings, frame limit, and Opus settings. Explicit codec flags must match
the saved profile; a mismatch reports the channel and differing fields. The
original numeric PSNR target is not saved: its selected concrete settings are
reused. Only an explicit `--target-psnr` reruns benchmark selection, and the result
must still match an existing profile.

Each channel keeps its own profile. `--procs` caps the invocation's total active
encoders, not workers per channel. Local INT16 jobs with matching profiles share
persistent workers; profile switches release that pool before loading another.
Machine-local settings include worker count (default 1), GPU (default 0), device,
decoder threads/prefetch, model paths, cookies and yt-dlp paths. No host paths or
credentials are imported from saved profiles.

Install the same Python project/environment on the helper, then copy the matching
UF prepared `*.int16.pt` files into its `checkpoints/`, or provide `--prepared_i`
and `--prepared_p`. Content identities and arithmetic are validated. Prepared
INT16 files are self-contained; float source checkpoints are not required to use
them. No weights are downloaded or prepared automatically by `join`. FP16 profiles
require matching local float checkpoints and CUDA. See [setup](UF_LOCAL_SETUP.md).

`join` and local `encode` process originals already present on disk. `stream`
keeps its remote input behavior. A batch snapshots discovered input entries;
newly added originals are picked up on the next invocation. All completed video
archives are checked first, including archives whose originals are absent.
Live peer claims remain pending, with a five-second poll; expired claims use the
existing recovery protocol (900-second default lease). Other available channels
are processed before waiting. Cancellation retains originals and committed
outputs. `join` always keeps originals and exits after the batch completes or
its failures have been reported; cleanup remains an explicit separate command.

Completed videos use `--resume-check fast` by default for `join`, `encode` and
`stream`. This validates the committed manifest, pipeline compatibility, required
audio/video, safe artifact paths, and each artifact's existence and recorded size.
An original that still exists must be a regular file with the recorded size and
modification time. Timestamps are compared exactly when both have fractional
seconds; when either is rounded to whole seconds (as with SSHFS/SFTP), their
whole-second values must match. Missing originals remain supported for completed
archives. These checks read metadata, not original or artifact payloads, avoiding
full-file transfers over network mounts. Peer completions and worker race checks
use the same policy. This local option is not part of the saved channel profile.

Use `--resume-check full` to hash every completed artifact and existing original
with SHA-256 as before. Full mode permits a changed original modification time
when its content hash still matches. Fast mode cannot detect same-size artifact
corruption or original content changes that also preserve the recorded mtime.
Neither resume mode performs neural decoding. Explicit `verify`, playback,
publication recovery and original deletion retain their full integrity checks;
fast resume never grants approval to delete originals.

The existing claim namespaces, pipeline identities, archive layouts and atomic
publication rules are unchanged. Helpers must share the physical output and
coordination directories. Their mount paths may differ.

## Names, resume and shared work

For a local `clip.mkv`, outputs are:

```text
clip_256x144_qI10_qP14.bin
clip.opus
clip.uf.json
clip.info.json                 (when provided)
clip.jpg_qI45.dcvci             (when thumbnail encoding is enabled)
```

A newly streamed video's basename is `VIDEO_ID_UNIX_TIMESTAMP`. The provider's
`timestamp` is preferred, followed by `release_timestamp`; date-only metadata
uses midnight UTC. Existing archive basenames are recognized on resume.
For duplicate local basenames, MKV is preferred, followed by MP4, WebM, MOV and
AVI. Alternate containers remain untouched. Relative input folders are retained.

Keep `.uf.json` with its video bitstream: it contains runtime, prepared-model
identities, frame counts, source identity, artifact hashes and the completion mode.
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
audio. Decoded output files are published only after decoding completes and available
recorded hashes are checked; existing output files are never replaced.

The shared RT Tk viewer provides Play/Pause, previous/next frame, a seek slider,
`--start_frame`, `--fps`, and display size limits. It decodes in memory without
writing reconstructed media. Seeking starts at the nearest preceding I-frame;
with `--intra_period -1`, a backward jump may require decoding from the start.
UF's padded final temporal packet is trimmed to the recorded frame count.
UF video viewing automatically loads archived Opus audio when available, for
both flat archives and legacy bundles. Play/Pause controls both streams; seeking
and frame stepping pause sound and reposition it for the next Play. Playback
follows the audio device's consumed-sample clock (with a small device-dependent
buffer delay); late video frames are skipped for display to avoid accumulating
drift. A shorter audio track lets video finish normally. `--fps` changes both
video speed and audio tempo while preserving pitch. No reconstructed media is
written, and the audio queue holds approximately a quarter-second of PCM.
`--no-audio` disables sound. Without audio, SDL2, FFmpeg or a working output
device, video remains playable; audio initialization/decoding errors appear in
the viewer status. Folder image viewing provides a gallery and reuses the intra
model between images. SDL2 is loaded from the system library; no Python audio
package is required.

UF `.dcvci` files are self-contained: they embed their pipeline, original image
size and payload hash. Older images also contain decoded-pixel validation. They use the UF intra network,
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

## Encode completion without a second neural pass

New video and intra-image encodes no longer run a full decoder before publication.
Their manifests explicitly record `validation.mode = "artifact-hashes"` and
`decode_performed = false`; they do not claim to contain a decoded-pixel baseline.
Atomic publication, source identity checks, artifact hashing and ownership checks
remain in place. This setting does not change the encoding pipeline fingerprint,
bitstreams, or existing archives, so a previous run can resume with the same args.

`verify` still decodes explicitly. Normal decode checks stream completeness and
model/runtime identities; if an older archive has a recorded decoded-pixel hash,
it also compares that hash. Cleanup still checks decoding before deleting a local
original. Nothing retroactively removes or updates an older validation record.
Use updated UF code to read new encode-only archives.

Streaming retains the full yt-dlp metadata document in `.info.json`, including formats, chapters, subtitles, categories and provider timestamps when available. It does not whitelist or discard fields.
