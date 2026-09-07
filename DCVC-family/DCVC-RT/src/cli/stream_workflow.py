from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

import src.cli.encode_workflow as encode_core
from src.cli.dashboard import EncodeDashboard
from src.cli.device_plan import build_worker_device_plan
from src.cli.progress import append_progress_log, ensure_dir


@dataclass(frozen=True)
class StreamTask:
    task_id: str
    video_id: str
    webpage_url: str
    channel_id: str
    extractor: str


@dataclass(frozen=True)
class RemoteMedia:
    metadata: Dict
    width: int
    height: int
    fps: Optional[float]
    has_audio: bool
    video_format_selector: str
    audio_format_selector: str


StreamInput = Union[str, StreamTask]

YOUTUBE_CHANNEL_ID_PATTERN = re.compile(r"^UC[A-Za-z0-9_-]{22}$")
YOUTUBE_VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")


def _safe_component(value: object, fallback: str, max_length: int = 160) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9._@+-]+", "_", text).strip("._")
    return (text or fallback)[:max_length]


def _cookies_args(cookies_path: Optional[str]) -> List[str]:
    return ["--cookies", cookies_path] if cookies_path else []


def _ytdlp_runtime_args(youtube_player_client: str) -> List[str]:
    arguments = [
        "--ignore-config",
        "--remote-components",
        "ejs:github",
    ]
    if youtube_player_client and youtube_player_client != "default":
        arguments.extend((
            "--extractor-args",
            f"youtube:player_client={youtube_player_client}",
        ))
    return arguments


def find_ytdlp_executable(requested: Optional[str] = None) -> str:
    """Find an external yt-dlp without requiring it in the active Conda env."""

    candidates: List[Optional[str]] = []
    if requested:
        requested_path = Path(requested).expanduser()
        if requested_path.parent != Path(".") or os.sep in requested:
            candidates.append(str(requested_path))
        else:
            candidates.append(shutil.which(requested))
    else:
        candidates.append(shutil.which("yt-dlp"))

    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        conda_bin = Path(conda_exe).expanduser().resolve().parent
        candidates.extend((str(conda_bin / "yt-dlp"), str(conda_bin / "yt-dlp.exe")))

    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())

    detail = f" at {requested!r}" if requested else ""
    raise RuntimeError(
        "yt-dlp executable not found"
        f"{detail}. Install yt-dlp anywhere on PATH, in the Conda base bin directory, "
        "or pass --yt-dlp /absolute/path/to/yt-dlp."
    )


def _run_ytdlp(executable: str, arguments: Sequence[str]) -> Tuple[int, str, str]:
    proc = encode_core.popen_command(
        [executable, *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = proc.communicate()
    return proc.returncode, stdout, stderr


def _entry_webpage_url(entry: Dict, source_url: str) -> Optional[str]:
    for key in ("webpage_url", "original_url", "url"):
        value = entry.get(key)
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return value

    video_id = str(entry.get("id") or "").strip()
    extractor = str(entry.get("extractor_key") or entry.get("extractor") or "").lower()
    if video_id and "youtube" in extractor:
        return f"https://www.youtube.com/watch?v={video_id}"
    if video_id and "twitch" in extractor:
        video_id = video_id[1:] if video_id[:1].lower() == "v" else video_id
        return f"https://www.twitch.tv/videos/{video_id}"
    if entry.get("_type") != "playlist":
        return source_url
    return None


def _entry_channel_id(entry: Dict, fallback: str) -> str:
    for key in (
        "channel_id",
        "playlist_channel_id",
        "uploader_id",
        "channel",
        "uploader",
        "playlist_id",
    ):
        value = entry.get(key)
        if value:
            return _safe_component(value, fallback)
    return fallback


def list_stream_tasks(
    executable: str,
    source_urls: Sequence[str],
    cookies_path: Optional[str] = None,
    max_videos: Optional[int] = None,
    youtube_player_client: str = "web_safari",
) -> Tuple[List[StreamTask], List[str]]:
    tasks: List[StreamTask] = []
    warnings: List[str] = []
    seen = set()

    for source_index, source_url in enumerate(source_urls, start=1):
        arguments = [
            *_ytdlp_runtime_args(youtube_player_client),
            "--flat-playlist",
            "--skip-download",
            "--no-warnings",
            "--no-progress",
            "--dump-json",
            *_cookies_args(cookies_path),
        ]
        if max_videos:
            arguments.extend(("--playlist-end", str(max_videos)))
        arguments.append(source_url)
        returncode, stdout, stderr = _run_ytdlp(executable, arguments)

        parsed = 0
        for line in stdout.splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(entry, dict):
                continue
            webpage_url = _entry_webpage_url(entry, source_url)
            if not webpage_url:
                continue
            video_id = _safe_component(entry.get("id"), f"item_{source_index}_{parsed + 1}")
            extractor = _safe_component(
                entry.get("extractor_key") or entry.get("extractor"),
                "generic",
            ).lower()
            task_extractor = "youtube" if "youtube" in extractor else extractor
            task_id = f"{task_extractor}:{video_id}"
            if task_id in seen:
                continue
            seen.add(task_id)
            fallback_channel = f"source_{source_index}"
            tasks.append(StreamTask(
                task_id=task_id,
                video_id=video_id,
                webpage_url=webpage_url,
                channel_id=_entry_channel_id(entry, fallback_channel),
                extractor=extractor,
            ))
            parsed += 1
            if max_videos and len(tasks) >= max_videos:
                break

        if returncode != 0:
            detail = stderr.strip().splitlines()[-1] if stderr.strip() else f"exit {returncode}"
            warnings.append(f"yt-dlp listing warning for {source_url}: {detail}")
        if parsed == 0:
            warnings.append(f"yt-dlp found no usable videos at {source_url}")
        if max_videos and len(tasks) >= max_videos:
            break

    return tasks, warnings


def _youtube_id_file_tasks(argument: str) -> List[StreamTask]:
    path = Path(argument).expanduser()
    channel_id = path.stem
    if path.suffix.lower() != ".txt" or not YOUTUBE_CHANNEL_ID_PATTERN.fullmatch(channel_id):
        raise ValueError(
            f"YouTube ID-list filename must be a channel ID followed by .txt: {argument}"
        )
    if not path.is_file():
        raise ValueError(f"YouTube ID-list file not found: {path}")

    tasks = []
    try:
        with path.open("r", encoding="utf-8-sig") as source:
            for line_number, line in enumerate(source, start=1):
                video_id = line.strip()
                if not video_id:
                    continue
                if not YOUTUBE_VIDEO_ID_PATTERN.fullmatch(video_id):
                    raise ValueError(
                        f"invalid YouTube video ID in {path} at line {line_number}: "
                        f"expected exactly 11 URL-safe characters"
                    )
                tasks.append(StreamTask(
                    task_id=f"youtube:{video_id}",
                    video_id=video_id,
                    webpage_url=f"https://www.youtube.com/watch?v={video_id}",
                    channel_id=channel_id,
                    extractor="youtube",
                ))
    except UnicodeDecodeError as exc:
        raise ValueError(f"YouTube ID-list file is not valid UTF-8: {path}") from exc
    if not tasks:
        raise ValueError(f"YouTube ID-list file contains no video IDs: {path}")
    return tasks


def build_stream_inputs(
    source_urls: Sequence[str],
    youtube_channels: Sequence[str],
    twitch_channels: Sequence[str],
) -> List[StreamInput]:
    inputs: List[StreamInput] = list(source_urls)
    for argument in youtube_channels:
        if argument.lower().endswith(".txt"):
            inputs.extend(_youtube_id_file_tasks(argument))
            continue
        if not YOUTUBE_CHANNEL_ID_PATTERN.fullmatch(argument):
            raise ValueError(
                "--youtube_channels entries must be a YouTube channel ID "
                f"(UC plus 22 URL-safe characters) or channel_ID.txt: {argument}"
            )
        inputs.append(f"https://www.youtube.com/channel/{argument}/videos")
    inputs.extend(
        f"https://www.twitch.tv/{channel}/videos?filter=archives&sort=time"
        for channel in twitch_channels
    )
    return inputs


def collect_stream_tasks(
    executable: str,
    inputs: Sequence[StreamInput],
    cookies_path: Optional[str] = None,
    max_videos: Optional[int] = None,
    youtube_player_client: str = "web_safari",
) -> Tuple[List[StreamTask], List[str]]:
    tasks: List[StreamTask] = []
    warnings: List[str] = []
    seen = set()

    for stream_input in inputs:
        if max_videos and len(tasks) >= max_videos:
            break
        if isinstance(stream_input, StreamTask):
            candidates = [stream_input]
        else:
            remaining = max_videos - len(tasks) if max_videos else None
            candidates, input_warnings = list_stream_tasks(
                executable,
                [stream_input],
                cookies_path=cookies_path,
                max_videos=remaining,
                youtube_player_client=youtube_player_client,
            )
            warnings.extend(input_warnings)
        for task in candidates:
            if task.task_id in seen:
                continue
            seen.add(task.task_id)
            tasks.append(task)
            if max_videos and len(tasks) >= max_videos:
                break

    return tasks, warnings


def _format_headers(value: object) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {
        str(name): str(header_value)
        for name, header_value in value.items()
        if header_value is not None
    }


def _parse_fps(value: object) -> Optional[float]:
    try:
        fps = float(value)
    except (TypeError, ValueError):
        return None
    return fps if fps > 0 else None


def resolve_remote_media(
    executable: str,
    task: StreamTask,
    max_height: Optional[int],
    cookies_path: Optional[str],
    youtube_player_client: str = "web_safari",
) -> RemoteMedia:
    height_filter = f"[height<={max_height}]" if max_height else ""
    metadata_format_selector = (
        f"bestvideo{height_filter}+bestaudio/"
        f"best{height_filter}/best"
    )
    video_format_selector = f"bestvideo{height_filter}/best{height_filter}/best"
    audio_format_selector = "bestaudio/best"
    arguments = [
        *_ytdlp_runtime_args(youtube_player_client),
        "--no-playlist",
        "--skip-download",
        "--no-warnings",
        "--no-progress",
        "--dump-single-json",
        "--format",
        metadata_format_selector,
        *_cookies_args(cookies_path),
        task.webpage_url,
    ]
    returncode, stdout, stderr = _run_ytdlp(executable, arguments)
    if returncode != 0:
        detail = stderr.strip()[-2000:] or f"exit {returncode}"
        raise RuntimeError(f"yt-dlp metadata/format resolution failed: {detail}")
    try:
        metadata = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("yt-dlp returned invalid metadata JSON") from exc
    if not isinstance(metadata, dict):
        raise RuntimeError("yt-dlp metadata result is not an object")

    selected_formats = metadata.get("requested_formats")
    if not isinstance(selected_formats, list) or not selected_formats:
        selected_formats = [metadata]

    video_format = next(
        (
            item for item in selected_formats
            if isinstance(item, dict)
            and item.get("vcodec") not in (None, "none")
        ),
        None,
    )
    audio_format = next(
        (
            item for item in selected_formats
            if isinstance(item, dict)
            and item.get("acodec") not in (None, "none")
        ),
        None,
    )
    if video_format is None:
        raise RuntimeError("yt-dlp did not resolve a playable video stream")

    return RemoteMedia(
        metadata=metadata,
        width=int(video_format.get("width") or metadata.get("width") or 0),
        height=int(video_format.get("height") or metadata.get("height") or 0),
        fps=_parse_fps(video_format.get("fps") or metadata.get("fps")),
        has_audio=audio_format is not None,
        video_format_selector=video_format_selector,
        audio_format_selector=audio_format_selector,
    )


def build_ytdlp_stream_command(
    executable: str,
    webpage_url: str,
    format_selector: str,
    cookies_path: Optional[str],
    youtube_player_client: str = "web_safari",
) -> List[str]:
    return [
        executable,
        *_ytdlp_runtime_args(youtube_player_client),
        "--no-playlist",
        "--no-warnings",
        "--no-progress",
        "--format",
        format_selector,
        *_cookies_args(cookies_path),
        "--output",
        "-",
        webpage_url,
    ]


def encode_remote_audio_to_temp(
    executable: str,
    webpage_url: str,
    format_selector: str,
    cookies_path: Optional[str],
    temporary_output: Path,
    opus_params: Dict,
    youtube_player_client: str = "web_safari",
):
    downloader = encode_core.popen_command(
        build_ytdlp_stream_command(
            executable,
            webpage_url,
            format_selector,
            cookies_path,
            youtube_player_client,
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    ffmpeg = None
    try:
        ffmpeg = encode_core.popen_command([
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-nostats",
            "-y",
            "-i",
            "pipe:0",
            "-vn",
            "-c:a",
            "libopus",
            "-b:a",
            str(opus_params["bitrate"]),
            "-vbr",
            str(opus_params["vbr"]),
            "-compression_level",
            str(opus_params["complexity"]),
            "-frame_duration",
            str(opus_params["frame_ms"]),
            "-ac",
            str(opus_params["channels"]),
            str(temporary_output),
        ], stdin=downloader.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if downloader.stdout:
            downloader.stdout.close()
        ffmpeg_returncode = ffmpeg.wait()
        downloader_returncode = downloader.wait()
        if downloader_returncode != 0:
            detail = _process_error_tail(downloader)
            raise RuntimeError(
                f"yt-dlp audio stream failed rc={downloader_returncode}{detail}"
            )
        if ffmpeg_returncode != 0:
            detail = _process_error_tail(ffmpeg)
            raise RuntimeError(f"ffmpeg Opus encode failed rc={ffmpeg_returncode}{detail}")
    finally:
        encode_core.kill_popen(ffmpeg)
        encode_core.kill_popen(downloader)


def _process_error_tail(process: Optional[subprocess.Popen], limit: int = 1200) -> str:
    if process is None or process.stderr is None:
        return ""
    try:
        output = process.stderr.read()
    except (OSError, ValueError):
        return ""
    if isinstance(output, bytes):
        output = output.decode("utf-8", "replace")
    lines = [line.strip() for line in str(output).splitlines() if line.strip()]
    if not lines:
        return ""
    return f": {' | '.join(lines[-3:])[-limit:]}"


def _atomic_write_json(path: Path, payload: Dict):
    ensure_dir(path.parent)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        encode_core.safe_unlink(temporary)


def _thumbnail_extension(metadata: Dict, url: str) -> str:
    for item in reversed(metadata.get("thumbnails") or []):
        if isinstance(item, dict) and item.get("url") == url:
            extension = str(item.get("ext") or "").lower()
            if extension in {"webp", "jpg", "jpeg", "png"}:
                return ".jpg" if extension == "jpeg" else f".{extension}"
    extension = Path(urllib.parse.urlparse(url).path).suffix.lower()
    return extension if extension in {".webp", ".jpg", ".jpeg", ".png"} else ".jpg"


def download_thumbnail(metadata: Dict, output_base: Path) -> Optional[Path]:
    url = metadata.get("thumbnail")
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return None
    destination = output_base.parent / (
        output_base.name + _thumbnail_extension(metadata, url)
    )
    if destination.exists():
        return destination

    headers = _format_headers(metadata.get("http_headers"))
    request = urllib.request.Request(url, headers=headers)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response, temporary.open("xb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        return destination
    finally:
        encode_core.safe_unlink(temporary)


def _existing_video_output(channel_out: Path, base: str, config: encode_core.EncoderCfg) -> Optional[Path]:
    matches = sorted(channel_out.glob(
        f"{base}_*x*_qI{config.qp_i}_qP{config.qp_p}.bin"
    ))
    return next((path for path in matches if path.is_file() and path.stat().st_size > 0), None)


def _completed_stream_output(
    channel_out: Path,
    video_id: str,
    config: encode_core.EncoderCfg,
    audio_enabled: bool,
) -> bool:
    if not channel_out.is_dir():
        return False
    for info_path in channel_out.glob(f"{video_id}_*.info.json"):
        base = info_path.name[:-len(".info.json")]
        video_path = _existing_video_output(channel_out, base, config)
        audio_path = channel_out / f"{base}.opus"
        if (
            video_path is not None
            and video_path.stat().st_size > 0
            and (not audio_enabled or (audio_path.is_file() and audio_path.stat().st_size > 0))
        ):
            return True
    return False


def process_stream_task(
    task: StreamTask,
    progress_q,
    worker_id: int,
    output_root: Path,
    encoder: encode_core.NeuralEncoder,
    config: encode_core.EncoderCfg,
    audio_enabled: bool,
    opus_params: Dict,
    finalize_mode: str,
    ytdlp_executable: str,
    cookies_path: Optional[str],
    source_max_height: Optional[int],
    write_thumbnail: bool,
    youtube_player_client: str = "web_safari",
) -> Tuple[bool, Optional[float]]:
    started = time.time()

    def emit(event_type: str, **fields):
        progress_q.put({
            "type": event_type,
            "wid": worker_id,
            "task_id": task.task_id,
            "vid": task.video_id,
            "channel_id": task.channel_id,
            "source_url": task.webpage_url,
            **fields,
        })

    emit("worker_task_start")
    try:
        initial_channel_out = output_root / task.channel_id
        if _completed_stream_output(
            initial_channel_out,
            task.video_id,
            config,
            audio_enabled,
        ):
            emit("worker_done", frames=0, elapsed=0.0)
            return True, 0.0

        media = resolve_remote_media(
            ytdlp_executable,
            task,
            source_max_height,
            cookies_path,
            youtube_player_client,
        )
        channel_id = _safe_component(
            media.metadata.get("channel_id") or task.channel_id,
            task.channel_id,
        )
        channel_out = output_root / channel_id
        ensure_dir(channel_out)
        upload_date = _safe_component(media.metadata.get("upload_date"), "00000000", 16)
        video_id = _safe_component(media.metadata.get("id"), task.video_id)
        base = f"{video_id}_{upload_date}"
        info_path = channel_out / f"{base}.info.json"
        _atomic_write_json(info_path, media.metadata)

        if write_thumbnail:
            try:
                download_thumbnail(media.metadata, channel_out / base)
            except Exception as exc:  # Thumbnail failure must not discard a valid encode.
                emit("log", message=f"thumbnail unavailable: {exc}")

        existing_bin = _existing_video_output(channel_out, base, config)
        opus_path = channel_out / f"{base}.opus"
        need_video = existing_bin is None
        need_audio = audio_enabled and not (
            opus_path.is_file() and opus_path.stat().st_size > 0
        )
        if not need_video and not need_audio:
            emit("worker_done", frames=0, elapsed=0.0, output_channel=channel_id)
            return True, 0.0
        if need_audio and not media.has_audio:
            raise RuntimeError("selected source has no playable audio stream")

        append_progress_log(channel_out, {
            "video_id": base,
            "remote_id": task.video_id,
            "status": "started",
            "source_url": task.webpage_url,
        })

        audio_thread = None
        audio_result = {"ok": not need_audio, "error": None}
        temporary_opus = None
        if need_audio:
            temporary_directory = Path(
                os.environ.get("ENC_RAM_TMP_DIR", encode_core.best_ram_dir())
            )
            temporary_opus = temporary_directory / f".tmp_{base}_{os.getpid()}.opus"
            encode_core.register_tmp_ram(temporary_opus)
            emit("worker_audio", status="started")

            def encode_audio():
                try:
                    encode_remote_audio_to_temp(
                        ytdlp_executable,
                        task.webpage_url,
                        media.audio_format_selector,
                        cookies_path,
                        temporary_opus,
                        opus_params,
                        youtube_player_client,
                    )
                    audio_result["ok"] = True
                except Exception as exc:  # pylint: disable=broad-except
                    audio_result["error"] = str(exc)
                    emit("worker_audio", status="failed")
                    append_progress_log(channel_out, {
                        "video_id": base,
                        "status": "failed",
                        "stage": "audio",
                        "error": str(exc),
                    })

            audio_thread = threading.Thread(
                target=encode_audio,
                name=f"stream-audio-{video_id}",
                daemon=True,
            )
            audio_thread.start()
        else:
            emit("worker_audio", status="done" if audio_enabled else "off")

        video_downloader = None
        ffmpeg_process = None
        staging_bin = None
        try:
            if need_video:
                source_width, source_height, source_fps = (
                    media.width,
                    media.height,
                    media.fps,
                )
                if source_width <= 0 or source_height <= 0:
                    source_width, source_height = 854, 480
                width, height = encode_core.compute_target_dims(
                    source_width,
                    source_height,
                    config.resolution,
                    config.pad_multiple,
                )
                output_bin = channel_out / (
                    f"{base}_{width}x{height}_qI{config.qp_i}_qP{config.qp_p}.bin"
                )
                staging_bin = channel_out / f".{output_bin.name}.stream.{os.getpid()}"
                encode_core.safe_unlink(staging_bin)
                video_downloader = encode_core.popen_command(
                    build_ytdlp_stream_command(
                        ytdlp_executable,
                        task.webpage_url,
                        media.video_format_selector,
                        cookies_path,
                        youtube_player_client,
                    ),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                ffmpeg_command = encode_core.build_ffmpeg_chain_local(
                    "pipe:0",
                    width,
                    height,
                    config,
                    source_width=source_width,
                    source_height=source_height,
                )
                resolution_text = f"{source_width}, {source_height} -> {width}, {height}"
                ffmpeg_process = encode_core.popen_command(
                    ffmpeg_command,
                    stdin=video_downloader.stdout,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                if video_downloader.stdout:
                    video_downloader.stdout.close()
                append_progress_log(channel_out, {
                    "video_id": base,
                    "status": "video-encoding",
                    "source_url": task.webpage_url,
                    "src_fps": source_fps,
                    "out_fps": config.fps or "source",
                    "source_width": source_width,
                    "source_height": source_height,
                    "output_width": width,
                    "output_height": height,
                    "resize_applied": (source_width, source_height) != (width, height),
                })
                emit(
                    "worker_start",
                    vid=base,
                    output_channel=channel_id,
                    audio="on" if need_audio else ("done" if audio_enabled else "off"),
                    resolution=resolution_text,
                )

                def on_progress(frames, fps, elapsed):
                    emit(
                        "worker_prog",
                        vid=base,
                        output_channel=channel_id,
                        frames=int(frames),
                        fps=float(fps),
                        elapsed=float(elapsed),
                    )

                encoder.encode_from_ffmpeg_rawpipe(
                    ffmpeg_process,
                    width,
                    height,
                    staging_bin,
                    log_dir=channel_out,
                    video_id=base,
                    on_progress=on_progress,
                    finalize_mode=finalize_mode,
                )
                if ffmpeg_process.stdout:
                    ffmpeg_process.stdout.close()
                ffmpeg_returncode = ffmpeg_process.wait(timeout=5)
                downloader_returncode = video_downloader.wait(timeout=5)
                if downloader_returncode != 0:
                    detail = _process_error_tail(video_downloader)
                    raise RuntimeError(
                        f"yt-dlp video stream failed rc={downloader_returncode}{detail}"
                    )
                if ffmpeg_returncode != 0:
                    detail = _process_error_tail(ffmpeg_process)
                    raise RuntimeError(
                        f"ffmpeg video stream failed rc={ffmpeg_returncode}{detail}"
                    )
                if not staging_bin.is_file() or staging_bin.stat().st_size <= 0:
                    raise RuntimeError("remote video stream produced no encoded frames")
                os.replace(staging_bin, output_bin)
                append_progress_log(channel_out, {
                    "video_id": base,
                    "status": "video-done",
                    "out_bin": str(output_bin),
                })

            if audio_thread:
                audio_thread.join()
            if need_audio:
                if not audio_result["ok"]:
                    raise RuntimeError(audio_result["error"] or "audio stream failed")
                encode_core.atomic_copy_across_fs(temporary_opus, opus_path)
                emit("worker_audio", status="done")

            elapsed = time.time() - started
            emit(
                "worker_done",
                vid=base,
                frames=None,
                elapsed=elapsed,
                output_channel=channel_id,
            )
            return True, elapsed
        finally:
            encode_core.kill_popen(ffmpeg_process)
            encode_core.kill_popen(video_downloader)
            if audio_thread and audio_thread.is_alive():
                encode_core.kill_all_children()
                audio_thread.join(timeout=5.0)
            if staging_bin is not None:
                encode_core.safe_unlink(staging_bin)
            if temporary_opus is not None:
                encode_core.safe_unlink(temporary_opus)
    except Exception as exc:  # pylint: disable=broad-except
        emit("worker_fail", stage="stream", error=str(exc))
        return False, None


def _stream_worker_entry(
    worker_id: int,
    task_queue,
    progress_queue,
    stop_event,
    output_root: str,
    model_i: str,
    model_p: str,
    config_dict: Dict,
    audio_mode: str,
    opus_params: Dict,
    use_cuda: bool,
    cuda_device_index: Optional[int],
    finalize_mode: str,
    ytdlp_executable: str,
    cookies_path: Optional[str],
    source_max_height: Optional[int],
    write_thumbnail: bool,
    youtube_player_client: str,
):
    encode_core.STOP_FLAG = False

    def handle_signal(received_signal, frame):
        del received_signal, frame
        encode_core.STOP_FLAG = True
        encode_core.kill_all_children()
        encode_core.cleanup_registered_ram_tmp()
        encode_core.release_cuda()
        raise SystemExit(130)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    encode_core.set_torch_env()
    if use_cuda and torch.cuda.is_available():
        cuda_device_index = 0 if cuda_device_index is None else cuda_device_index
        torch.cuda.set_device(cuda_device_index)
        device = torch.device(f"cuda:{cuda_device_index}")
    else:
        device = torch.device("cpu")

    encoder = encode_core.NeuralEncoder(
        device,
        model_i,
        model_p,
        encode_core.EncoderCfg(**config_dict),
    )
    progress_queue.put({
        "type": "worker_hello",
        "wid": worker_id,
        "pid": os.getpid(),
        "device": str(device),
    })

    while not stop_event.is_set():
        try:
            task = task_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if task is None:
            break
        process_stream_task(
            task=task,
            progress_q=progress_queue,
            worker_id=worker_id,
            output_root=Path(output_root),
            encoder=encoder,
            config=encode_core.EncoderCfg(**config_dict),
            audio_enabled=audio_mode == "opus",
            opus_params=opus_params,
            finalize_mode=finalize_mode,
            ytdlp_executable=ytdlp_executable,
            cookies_path=cookies_path,
            source_max_height=source_max_height,
            write_thumbnail=write_thumbnail,
            youtube_player_client=youtube_player_client,
        )

    encode_core.kill_all_children()
    encode_core.cleanup_registered_ram_tmp()
    encode_core.release_cuda()


def configure_parser(parser: argparse.ArgumentParser):
    source_group = parser.add_argument_group("Remote sources")
    source_group.add_argument(
        "--source_urls",
        nargs="+",
        default=None,
        help="yt-dlp-supported video, playlist, or channel URLs.",
    )
    source_group.add_argument(
        "--youtube_channels",
        nargs="+",
        default=None,
        help=(
            "YouTube channel IDs (UC...) and/or channel_ID.txt files containing "
            "one video ID per line."
        ),
    )
    source_group.add_argument(
        "--twitch_channels",
        nargs="+",
        default=None,
        help="Twitch channel names whose archived broadcasts should be streamed.",
    )
    source_group.add_argument("--cookies", default=None, help="Optional Netscape cookies file for yt-dlp.")
    source_group.add_argument(
        "--yt-dlp",
        dest="ytdlp_path",
        default=None,
        help="External yt-dlp executable. Auto-detected from PATH or the Conda base bin directory.",
    )
    source_group.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Maximum total videos to queue, useful for a test run.",
    )
    source_group.add_argument(
        "--source-max-height",
        type=int,
        default=480,
        help="Maximum selected source height before neural resizing; use 0 for no limit.",
    )
    source_group.add_argument(
        "--youtube-player-client",
        default="web_safari",
        help="yt-dlp YouTube player client; use 'default' to let yt-dlp choose.",
    )
    source_group.add_argument(
        "--no-thumbnail",
        action="store_false",
        dest="write_thumbnail",
        help="Do not retain the source thumbnail beside the encoded artifacts.",
    )
    source_group.set_defaults(write_thumbnail=True)

    io_group = parser.add_argument_group("Output")
    io_group.add_argument("--output_root", required=True)

    hardware_group = parser.add_argument_group("Hardware and concurrency")
    hardware_group.add_argument("--procs", type=int, default=None)
    hardware_group.add_argument(
        "--cuda",
        type=lambda value: value.lower() in {"1", "true", "yes"},
        default=True,
    )
    hardware_group.add_argument("--cuda_idx", nargs="+", type=int, default=None)

    video_group = parser.add_argument_group("Neural video encoder")
    video_group.add_argument("--model_path_i", default="./checkpoints/cvpr2025_image.pth.tar")
    video_group.add_argument("--model_path_p", default="./checkpoints/cvpr2025_video.pth.tar")
    video_group.add_argument("--qp_i", type=int, default=32)
    video_group.add_argument("--qp_p", type=int, default=10)
    video_group.add_argument("--resolution", type=int, default=96)
    video_group.add_argument("--pad_multiple", type=int, default=16)
    video_group.add_argument("--force_intra_period", type=int, default=-1)
    video_group.add_argument("--reset_interval", type=int, default=32)
    video_group.add_argument("--force_zero_thres", type=float, default=None)

    audio_group = parser.add_argument_group("Audio (Opus)")
    audio_group.add_argument("--audio", choices=["opus", "none"], default="opus")
    audio_group.add_argument("--opus_bitrate", default="12k")
    audio_group.add_argument("--opus_frame_ms", type=float, default=60.0)
    audio_group.add_argument("--opus_complexity", type=int, default=10)
    audio_group.add_argument("--opus_channels", choices=["mono", "stereo"], default="stereo")
    audio_group.add_argument("--opus_vbr", choices=["on", "off", "constrained"], default="on")

    pipeline_group = parser.add_argument_group("Pipeline and FFmpeg")
    pipeline_group.add_argument("--fps", type=float, default=30)
    pipeline_group.add_argument(
        "--ff_color_matrix",
        choices=["bt709", "bt601", "bt2020"],
        default="bt709",
    )
    pipeline_group.add_argument("--ff_hwaccel", choices=["none", "auto"], default="none")
    pipeline_group.add_argument("--ffmpeg_prefetch", type=int, default=8)
    pipeline_group.add_argument("--ram_tmp_dir", default=None)
    pipeline_group.add_argument("--disk_finalize", choices=["direct", "atomic"], default="atomic")
    pipeline_group.add_argument("--ui", choices=["auto", "tui", "plain"], default="auto")


def _channel_snapshots(channel_state: Dict[str, Dict]) -> List[Dict]:
    snapshots = []
    for channel_id, state in sorted(channel_state.items()):
        processed = state["processed"]
        failures = state["failures"]
        total = state["total"]
        if processed < total:
            status = "encoding"
        elif failures:
            status = "stream-failed"
        else:
            status = "complete"
        snapshots.append({
            "channel_id": channel_id,
            "status": status,
            "processed": processed,
            "pending": total,
            "already_done": 0,
            "failures": failures,
            "active": state["active"],
            "total": total,
            "reclaimable_bytes": 0,
            "last_error": state.get("last_error", ""),
        })
    return snapshots


def run(args) -> int:
    try:
        stream_inputs = build_stream_inputs(
            args.source_urls or [],
            args.youtube_channels or [],
            args.twitch_channels or [],
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return 2
    if not stream_inputs:
        print("Error: provide --source_urls, --youtube_channels, and/or --twitch_channels.")
        return 2
    if args.max_videos is not None and args.max_videos < 1:
        print("Error: --max-videos must be at least 1.")
        return 2
    if args.source_max_height is not None and args.source_max_height < 0:
        print("Error: --source-max-height cannot be negative.")
        return 2
    if args.cookies:
        cookies_path = Path(args.cookies).expanduser()
        if not cookies_path.is_file():
            print(f"Error: cookies file not found: {cookies_path}")
            return 2
        args.cookies = str(cookies_path.resolve())

    try:
        ytdlp_executable = find_ytdlp_executable(args.ytdlp_path)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 2
    try:
        version = subprocess.run(
            [ytdlp_executable, "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"Error: cannot run yt-dlp at {ytdlp_executable}: {exc}")
        return 2
    print(f"yt-dlp: {ytdlp_executable} ({version})")

    tasks, listing_warnings = collect_stream_tasks(
        ytdlp_executable,
        stream_inputs,
        cookies_path=args.cookies,
        max_videos=args.max_videos,
        youtube_player_client=args.youtube_player_client,
    )
    for warning in listing_warnings:
        print(f"Warning: {warning}")
    if not tasks:
        print("Nothing to do — yt-dlp found no videos.")
        return 0

    try:
        device_plan = build_worker_device_plan(
            use_cuda=args.cuda,
            requested_workers=args.procs,
            requested_cuda_indices=args.cuda_idx,
            cuda_available=torch.cuda.is_available(),
            cuda_device_count=torch.cuda.device_count(),
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return 2
    if args.cuda and not device_plan.using_cuda:
        print("Warning: --cuda specified but no devices found. Running on CPU.")

    output_root = Path(args.output_root).expanduser()
    ensure_dir(output_root)
    output_root = output_root.resolve()
    ram_directory = encode_core.best_ram_dir(args.ram_tmp_dir)
    os.environ["ENC_RAM_TMP_DIR"] = str(ram_directory)

    config_dict = {
        "qp_i": args.qp_i,
        "qp_p": args.qp_p,
        "force_intra_period": args.force_intra_period,
        "reset_interval": args.reset_interval,
        "resolution": args.resolution,
        "force_zero_thres": args.force_zero_thres,
        "pad_multiple": args.pad_multiple,
        "ff_color_matrix": args.ff_color_matrix,
        "ff_hwaccel": args.ff_hwaccel,
        "fps": args.fps,
        "ffmpeg_prefetch": args.ffmpeg_prefetch,
    }
    opus_params = {
        "bitrate": args.opus_bitrate,
        "frame_ms": args.opus_frame_ms,
        "complexity": args.opus_complexity,
        "channels": 1 if args.opus_channels == "mono" else 2,
        "vbr": args.opus_vbr,
    }
    source_max_height = args.source_max_height or None

    worker_count = min(device_plan.worker_count, len(tasks))
    cuda_plan = device_plan.cuda_indices[:worker_count]
    if device_plan.using_cuda:
        description = ", ".join(
            f"worker {worker_id}->cuda:{cuda_plan[worker_id]}"
            for worker_id in range(worker_count)
        )
    else:
        noun = "worker" if worker_count == 1 else "workers"
        description = f"{worker_count} CPU {noun}"
    print(f"Worker plan: {description}")
    print("Source media is streamed; no original video container will be written.")

    context = get_context("spawn")
    task_queue = context.Queue()
    progress_queue = context.Queue()
    stop_event = context.Event()
    workers = []
    for worker_id in range(worker_count):
        worker = context.Process(
            target=_stream_worker_entry,
            args=(
                worker_id,
                task_queue,
                progress_queue,
                stop_event,
                str(output_root),
                args.model_path_i,
                args.model_path_p,
                config_dict,
                args.audio,
                opus_params,
                device_plan.using_cuda,
                cuda_plan[worker_id],
                args.disk_finalize,
                ytdlp_executable,
                args.cookies,
                source_max_height,
                args.write_thumbnail,
                args.youtube_player_client,
            ),
            daemon=True,
        )
        worker.start()
        workers.append(worker)

    for task in tasks:
        task_queue.put(task)
    for _ in workers:
        task_queue.put(None)

    progress = EncodeDashboard(len(tasks), len(workers), output_root, mode=args.ui)
    expected = {task.task_id for task in tasks}
    terminal = set()
    active: Dict[int, Dict] = {}
    device_labels = {
        worker_id: (
            f"cuda:{cuda_plan[worker_id]}"
            if cuda_plan[worker_id] is not None
            else "cpu"
        )
        for worker_id in range(worker_count)
    }
    channel_state: Dict[str, Dict] = {}
    for task in tasks:
        state = channel_state.setdefault(
            task.channel_id,
            {"total": 0, "processed": 0, "failures": 0, "active": 0},
        )
        state["total"] += 1
    events: List[str] = []
    dead_since = None
    interrupted = False
    unreported_failure = False

    try:
        while True:
            try:
                message = progress_queue.get(timeout=0.1)
            except queue.Empty:
                message = None

            if message:
                message_type = message.get("type")
                worker_id = message.get("wid", 0)
                task_id = message.get("task_id")
                channel_id = message.get("channel_id")
                video_id = message.get("vid", "-")
                if message_type == "worker_hello":
                    progress.set_worker_text(
                        worker_id,
                        f"{message.get('device', 'unknown')} ready (pid {message.get('pid', '?')})",
                    )
                elif message_type == "worker_task_start":
                    active[worker_id] = {
                        "vid": video_id,
                        "device": device_labels[worker_id],
                        "frames": 0,
                        "fps": 0.0,
                        "audio": "resolving",
                    }
                    if channel_id in channel_state:
                        channel_state[channel_id]["active"] += 1
                    progress.set_worker_text(worker_id, encode_core._format_worker_text(active[worker_id]))
                elif message_type == "worker_start":
                    state = active.setdefault(worker_id, {"device": device_labels[worker_id]})
                    state.update({
                        "vid": video_id,
                        "frames": 0,
                        "fps": 0.0,
                        "audio": message.get("audio", "off"),
                        "resolution": message.get("resolution", ""),
                    })
                    progress.set_worker_text(worker_id, encode_core._format_worker_text(state))
                elif message_type == "worker_prog":
                    state = active.setdefault(worker_id, {"device": device_labels[worker_id]})
                    state.update({
                        "vid": video_id,
                        "frames": message.get("frames", 0),
                        "fps": message.get("fps", 0.0),
                    })
                    progress.set_worker_text(worker_id, encode_core._format_worker_text(state))
                elif message_type == "worker_audio":
                    state = active.setdefault(worker_id, {"device": device_labels[worker_id]})
                    state["audio"] = message.get("status", "off")
                    progress.set_worker_text(worker_id, encode_core._format_worker_text(state))
                elif message_type in {"worker_done", "worker_fail"} and task_id not in terminal:
                    terminal.add(task_id)
                    active.pop(worker_id, None)
                    progress.clear_worker(worker_id)
                    progress.increment_done(
                        elapsed_seconds=(
                            message.get("elapsed") if message_type == "worker_done" else None
                        )
                    )
                    if channel_id in channel_state:
                        state = channel_state[channel_id]
                        state["processed"] += 1
                        state["active"] = max(0, state["active"] - 1)
                        if message_type == "worker_fail":
                            state["failures"] += 1
                            state["last_error"] = message.get("error", "stream failed")
                    if message_type == "worker_fail":
                        events.append(f"{channel_id}/{video_id}: {message.get('error', 'stream failed')}")
                elif message_type == "log":
                    events.append(f"{channel_id}/{video_id}: {message.get('message', '')}")

            progress.update_lifecycle(_channel_snapshots(channel_state), events)
            alive = any(worker.is_alive() for worker in workers)
            if alive:
                dead_since = None
            elif terminal != expected:
                if dead_since is None:
                    dead_since = time.monotonic()
                elif time.monotonic() - dead_since >= 1.0:
                    missing = expected - terminal
                    for task_id in missing:
                        task = next(task for task in tasks if task.task_id == task_id)
                        terminal.add(task_id)
                        progress.increment_done()
                        events.append(f"{task_id}: workers exited without reporting completion")
                        state = channel_state[task.channel_id]
                        state["processed"] += 1
                        state["failures"] += 1
                        state["active"] = max(0, state["active"] - 1)
                        state["last_error"] = "worker exited without reporting completion"
                        unreported_failure = True
            if not alive and terminal == expected:
                break
    except KeyboardInterrupt:
        interrupted = True
        stop_event.set()
        progress.write("Interrupted. Stopping stream workers and discarding partial outputs...")
        encode_core._terminate_workers(workers)
    finally:
        progress.close()
        stop_event.set()
        if interrupted:
            encode_core._terminate_workers(workers, grace_seconds=1.0)
        else:
            for worker in workers:
                worker.join()
        encode_core._close_queue(task_queue)
        encode_core._close_queue(progress_queue)
        encode_core.kill_all_children()
        encode_core.cleanup_registered_ram_tmp()

    failures = sum(state["failures"] for state in channel_state.values())
    if interrupted:
        return 130
    print(f"Streaming encode finished: {len(terminal) - failures} succeeded, {failures} failed.")
    return 1 if failures or unreported_failure or terminal != expected else 0
