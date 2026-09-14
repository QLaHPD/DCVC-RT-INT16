# UF archives

The root `main.py` provides encode, decode, verify, view, and cleanup commands.
The default is UF FP16 inference with HT-S. Add `--runtime int16` for prepared
integer inference; decode, verify, view and cleanup select the manifest runtime
automatically. See [integer setup and portability](UF_INT16.md).
The FP16 HT-S path has passed native Jetson encode/decode validation; see
[the Bunny results](UF_BUNNY_VALIDATION.md). FP16 HT-L and LD have not yet been
validated with real bitstreams in this archive CLI.

## Encode

From the UF checkout:

```bash
./scripts/uf-python main.py encode --input_file Big_Buck_Bunny_720_10s_30MB.mp4 --output_root runs/bunny-qi36-qp30 --qi 36 --qp 30 --procs 1
```

Without `--resolution` or `--fps`, the original dimensions and frame rate are
used (odd dimensions are made even for YUV420). `--resolution 144` selects the
short edge and preserves aspect ratio; 1280×720 becomes 256×144. Scaling is
omitted when the dimensions already match. `--fps 8` selects eight frames per
second. Frame timestamps are normalized to constant frame rate.

Folder input uses `--base_root PATH` instead of `--input_file`. Optional
`--channel_ids ID1 ID2` limits discovery to those subfolders. Relative folders
are retained under `--output_root`. `--procs` controls parallel videos;
`--cuda_idx 0 1` assigns workers across GPUs. **Use one worker on this Jetson:**
two simultaneous HT-S workers exceeded its RAM during validation, even at 144p.
Failed jobs retain their originals and can be retried with fewer workers.
`--qp_i` and `--qp_p` are aliases for `--qi` and `--qp`.
`--skip_thres` defaults to zero. A positive value forces latent residuals with
predicted scale at or below the threshold to zero and omits them from entropy
coding. This can reduce size and change reconstruction quality. The value is
stored in the archive pipeline so decode and verification use the same setting.
HT-S and HT-L process temporal chunks of eight frames, so their positive
`--intra_period` must be `1` or a multiple of eight; LD accepts any positive
period.

INT16 input decoding defaults to up to four threads per worker and
`--prefetch_frames 8`. The thread budget uses the process's available CPUs,
divides them among `--procs`, reserves one CPU per worker for neural dispatch,
and clamps decoder threads to 1–4. An explicit `--input_threads` overrides it.
FP16 retains its previous defaults of one decoder thread and no read-ahead.
The bounded read-ahead queue overlaps CPU input decoding with neural encoding.
Queued YUV444 arrays are capped at 8 MiB, or one frame if a single frame exceeds
that budget; a producer may hold one additional frame. The encoder's own chunk
buffers and FFmpeg buffers are separate. Use `--input_threads 1
--prefetch_frames 0` to minimize input CPU/memory use. These performance settings
do not change archive compatibility or force re-encoding of completed videos.

`--model_structure hts|htl|ld` selects the model. Checkpoints default to
`checkpoints/cvpr2026_image.pth.tar` and the corresponding video checkpoint.
`--model_path_i` and `--model_path_p` override them.

New runs write flat sibling files in the channel/output folder: `<video>.bin`,
`<video>.opus` (when audio exists), and `<video>.uf.json`, plus metadata/thumbnail
sidecars. Keep the manifest with the bitstream: it records model identities,
playback settings, artifact hashes, source identity and decode validation.
Duplicate local stems retain their source extension in the output name to avoid
collisions (for example `clip.mkv.bin` and `clip.mp4.bin`).

Older `<filename>.uf/` bundles remain readable and are recognized on resume;
updating code does not move existing archives. An already-running process keeps
its original layout. New jobs carry the flat-layout setting explicitly.

Payloads are decoded and hashed in a hidden staging directory. Flat payloads are
published without overwriting existing files, followed by the manifest as the
completion record. Interrupted publication can reuse only byte-identical files;
a differing final output is never overwritten. Filesystem leases coordinate
instances using identical settings. Keep separate output directories for different
configurations; UF never replaces an RT bitstream sharing its filename.

Audio defaults to mono Opus at 6 kbps. `--audio none` omits audio, which prevents
original deletion if that source had audio. Timestamped events are also appended
to `.uf-progress.jsonl`.

`--ui plain` prints events. `--ui tui` shows frame progress/FPS, channel inventory,
and cleanup controls; `--ui auto` selects TUI in an interactive terminal and plain
otherwise. A TUI run stays open for review after encoding, including a run where
all videos were already encoded. Arrows select a channel, **d** requests deletion
of its verified originals, **a** requests deletion for all eligible listed channels,
**y** confirms, and **q** exits. Declining a prompt leaves originals untouched.

Append `--auto-delete` to encode to verify and delete matching local originals
automatically after the run. Failed/busy channels are excluded. `--keep-originals`
retains originals and exits without a cleanup prompt. `--cleanup-dry-run` previews
exact paths without deleting. These policies do not change bitstream compatibility.

## Decode and verify

```bash
./scripts/uf-python main.py verify --input runs/bunny-qi36-qp30/Big_Buck_Bunny_720_10s_30MB.bin
./scripts/uf-python main.py decode --input runs/bunny-qi36-qp30/Big_Buck_Bunny_720_10s_30MB.bin --output_file runs/bunny-reconstructed.mkv
./scripts/uf-python main.py view --input runs/bunny-qi36-qp30/Big_Buck_Bunny_720_10s_30MB.bin
```

The input accepts a flat `.bin` or `.uf.json`, or an older bundle directory / `video.bin`. Decode writes lossless
FFV1 reconstruction in Matroska and copies archived audio. View pipes video
frames to ffplay without storing decoded media; this initial viewer has no
seeking or synchronized audio playback. Both require the matching checkpoints.
Decoded pixels are compared with the recorded validation hash. FP16 decoding
across different GPUs is not guaranteed to reproduce that hash.

## Explicit original cleanup

```bash
./scripts/uf-python main.py cleanup --input PATH/VIDEO.bin
./scripts/uf-python main.py cleanup --input PATH/VIDEO.bin --yes
```

The first command previews the exact original file. `--yes` verifies decoding
and source hashes before deleting only that video. Sidecars, folders and other
files are retained. Partial encodes (`--max_frames`) and archives omitting
source audio cannot authorize deletion. `--base_root` remaps the original's
relative path when accessing the archive on another machine.

To review a channel independently of encoding (including legacy bundles):

```bash
./scripts/uf-python main.py manage --output_root /mnt/to_storage/DATA/YOUTUBE --channel_ids CHANNEL_ID --ui tui
./scripts/uf-python main.py manage --output_root /mnt/to_storage/DATA/YOUTUBE --channel_ids CHANNEL_ID --ui plain --auto-delete
```

`manage` reviews all archives in the selected output folders. An encode run limits
cleanup to its own discovered sources. Before deleting, cleanup verifies artifact
hashes, complete decode validation and the exact original's hash; only the original
video is removed. Other files and directories are retained. Leases serialize
cleanup with an encoder/other cleanup instance. TUI inventory refreshes show when
another instance has removed originals.

YouTube input uses `stream --source_urls URL` with the same encode settings; see
[the streaming example](../README.md#youtube-streaming). Streamed archives have no
stored original video and are shown as such, so neither automatic cleanup nor TUI
confirmation deletes anything for them. Image archive commands are not yet exposed
by this UF CLI.
