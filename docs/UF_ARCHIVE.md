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

INT16 input decoding uses `--input_threads 2` and `--prefetch_frames 8` by default.
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

Each source creates a `<filename including extension>.uf` directory containing
`video.bin`, a manifest, optional Opus audio, and available metadata/thumbnail
sidecars. Keep the whole bundle: UF bitstreams do not store every playback or
model parameter. Audio defaults to mono Opus at 6k; use `--opus_channels stereo`
and a suitable `--opus_bitrate` when desired. `--audio none` omits audio.

Archives are decoded and hashed before atomic publication. Repeating the same
command checks and resumes existing bundles. Different configurations require a
separate output directory. Filesystem leases coordinate instances sharing the
same output directory; `--shared-work --shared-instance NAME` is accepted.
Use identical checkpoints and settings on every instance. Busy work is reported
and skipped by that invocation. Events include timestamps and are also appended
to `.uf-progress.jsonl`. `--ui auto` and `--ui plain` currently both print events.

## Decode and verify

```bash
./scripts/uf-python main.py verify --input runs/bunny-qi36-qp30/Big_Buck_Bunny_720_10s_30MB.mp4.uf
./scripts/uf-python main.py decode --input runs/bunny-qi36-qp30/Big_Buck_Bunny_720_10s_30MB.mp4.uf --output_file runs/bunny-reconstructed.mkv
./scripts/uf-python main.py view --input runs/bunny-qi36-qp30/Big_Buck_Bunny_720_10s_30MB.mp4.uf
```

The input can also point to the bundle's `video.bin`. Decode writes lossless
FFV1 reconstruction in Matroska and copies archived audio. View pipes video
frames to ffplay without storing decoded media; this initial viewer has no
seeking or synchronized audio playback. Both require the matching checkpoints.
Decoded pixels are compared with the recorded validation hash. FP16 decoding
across different GPUs is not guaranteed to reproduce that hash.

## Explicit original cleanup

```bash
./scripts/uf-python main.py cleanup --input PATH/VIDEO.mp4.uf
./scripts/uf-python main.py cleanup --input PATH/VIDEO.mp4.uf --yes
```

The first command previews the exact original file. `--yes` verifies decoding
and source hashes before deleting only that video. Sidecars, folders and other
files are retained. Partial encodes (`--max_frames`) and archives omitting
source audio cannot authorize deletion. `--base_root` remaps the original's
relative path when accessing the archive on another machine.

This initial UF CLI does not yet provide the RT TUI, URL downloading, or image
archive commands. Original videos are retained by every encode command.
