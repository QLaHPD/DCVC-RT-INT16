# Direct remote streaming

`main.py stream` resolves remote sources with yt-dlp and sends their media streams directly through FFmpeg to the existing encoder. It never asks yt-dlp to download or merge a source container.

## Data path

```text
remote -> yt-dlp stdout -> FFmpeg -> bounded YUV RAM queue -> INT16 encoder -> .bin
remote -> yt-dlp stdout -> FFmpeg/libopus -> RAM temporary -> atomic commit -> .opus
remote -> yt-dlp metadata -------------------------------------------> .info.json
thumbnail URL --------------------------------------------------------> image sidecar
```

Only the final archival artifacts are stored. Raw frames are bounded by `--ffmpeg_prefetch`; the encoded DCVC bitstream is committed atomically. Audio uses `--ram_tmp_dir`, then copies atomically to its final path. If `/dev/shm` is writable it is selected automatically.

## yt-dlp discovery

yt-dlp does not need to be installed in the DCVC Conda environment. Discovery checks:

1. `--yt-dlp /explicit/path`
2. `yt-dlp` on `PATH`
3. `yt-dlp` beside the executable named by `CONDA_EXE`

Install or update yt-dlp separately on each computer. The command exits before loading models if the executable cannot be found or run.

For current YouTube extraction, the command allows yt-dlp to obtain its recommended EJS challenge component from GitHub and selects the `web_safari` player client. This avoids media-download failures observed with yt-dlp's automatic Android VR selection on the development host. Override it with `--youtube-player-client CLIENT`, or use `--youtube-player-client default` to restore yt-dlp's automatic choice.

## Sources

General yt-dlp URL:

```bash
DCVC_USE_INT16=1 python main.py stream \
  --source_urls 'https://example.invalid/video-or-playlist' \
  --output_root /data/encoded
```

YouTube channels:

```bash
DCVC_USE_INT16=1 python main.py stream \
  --youtube_channels \
    UCxxxxxxxxxxxxxxxxxxxxxx \
    ./UCyyyyyyyyyyyyyyyyyyyyyy.txt \
    ./UCzzzzzzzzzzzzzzzzzzzzzz.txt \
  --output_root /data/encoded
```

Each `--youtube_channels` value may be either a direct channel ID or a text-file path, in any combination. A text file must be named `<channel-ID>.txt`, where the channel ID is `UC` followed by 22 URL-safe characters. It contains one 11-character YouTube video ID per line; blank lines are ignored. URLs and additional columns are rejected so malformed lists fail before encoder workers start. Duplicate video IDs across files, direct channels, playlists, and other source inputs are queued once, with the first occurrence determining queue order.

Twitch archives:

```bash
DCVC_USE_INT16=1 python main.py stream \
  --twitch_channels channel_name \
  --output_root /data/encoded
```

The three source options may be combined. `--max-videos 1` is recommended for the first test. Use `--cookies` for sources that require an authenticated yt-dlp session.

## Formats and bandwidth

The default selector requests the best separate video and audio streams at or below `--source-max-height 480`, with a combined-format fallback. The neural encoder then resizes video according to `--resolution`. Set `--source-max-height 0` to remove the download-height limit.

Each worker starts yt-dlp immediately before use and connects its standard output directly to FFmpeg. This leaves protocol details, redirects, request headers, and expiring media URLs under yt-dlp's control. On a multi-GPU computer, each worker streams and encodes one complete video independently.

## Outputs and restart behavior

Outputs use the basename `<source-id>_<upload-date>` inside a channel directory:

```text
<id>_<date>_<width>x<height>_qI<qp>_qP<qp>.bin
<id>_<date>.opus
<id>_<date>.info.json
<id>_<date>.<thumbnail-extension>
```

An item is skipped when its metadata, non-empty matching bitstream, and requested non-empty Opus output already exist. A video-only success followed by an audio failure can therefore retry just the audio on the next run.

Because there is no local source file, the local-source deletion lifecycle is not invoked by `stream`.
