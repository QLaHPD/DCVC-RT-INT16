"""Metadata-only video indexing and bounded, on-demand FFmpeg clip decoding."""

from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import torch


def check_source(record):
    path = Path(record["source"])
    stat = path.stat()
    if [stat.st_size, stat.st_mtime_ns] != record["source_stat"]:
        raise ValueError(f"video changed since manifest creation: {path}")


def index_video(path, short_edge):
    path = Path(path).resolve()
    stat = path.stat()
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,nb_frames,avg_frame_rate,duration:format=duration",
         "-of", "json", str(path)], capture_output=True, text=True, check=True, timeout=60)
    metadata = json.loads(result.stdout)
    if not metadata.get("streams"):
        raise ValueError(f"no video stream: {path}")
    stream = metadata["streams"][0]
    try:
        fps = float(Fraction(stream["avg_frame_rate"]))
        duration = float(stream.get("duration") or metadata.get("format", {}).get("duration"))
        width, height = int(stream["width"]), int(stream["height"])
        count = stream.get("nb_frames", "N/A")
        count = int(count) if count != "N/A" else math.floor(duration * fps)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError(f"invalid video timing metadata: {path}") from exc
    if not math.isfinite(fps) or not math.isfinite(duration) or min(fps, duration, width, height) <= 0 or count < 2:
        raise ValueError(f"video needs valid timing, dimensions and at least two frames: {path}")
    filters = []
    if short_edge is not None:
        ratio = short_edge / min(width, height)
        width, height = max(2, round(width * ratio / 2) * 2), max(2, round(height * ratio / 2) * 2)
        filters.append(f"scale={width}:{height}:flags=bicubic")
    elif width % 2 or height % 2:
        width, height = width + width % 2, height + height % 2
        filters.append(f"pad={width}:{height}")
    record = {"source": str(path), "source_stat": [stat.st_size, stat.st_mtime_ns],
              "width": width, "height": height, "frame_count": count, "fps": fps,
              "duration": duration, "video_filters": filters}
    check_source(record)
    return record


def load_video_clip(record, start, count):
    """Seek by timestamp, then return consecutive decoded frames (no resampling).

    Frame counts/average FPS are metadata estimates for variable-rate media.
    A short tail retries an earlier timestamp, then the beginning, deterministically.
    Clips are never padded with repeated frames or replaced by another source.
    """
    check_source(record)
    width, height = record["width"], record["height"]
    pixels = width * height
    frame_bytes = pixels * 3 // 2
    timestamp = min(start / record["fps"], max(0, record["duration"] - count / record["fps"]))
    attempts = dict.fromkeys((timestamp, max(0, timestamp - count / record["fps"] - 1), 0.0))
    for seek in attempts:
        command = ["ffmpeg", "-v", "error", "-nostdin", "-threads", "1",
                   "-ss", f"{seek:.9f}", "-noautorotate", "-i", record["source"],
                   "-map", "0:v:0", "-an", "-sn", "-dn", "-vsync", "0",
                   "-filter_threads", "1"]
        if record["video_filters"]:
            command.extend(["-vf", ",".join(record["video_filters"])])
        command.extend(["-frames:v", str(count), "-pix_fmt", "yuv420p", "-threads", "1",
                        "-f", "rawvideo", "pipe:1"])
        result = subprocess.run(command, capture_output=True, timeout=120)
        if result.returncode:
            raise ValueError(f"FFmpeg clip decode failed for {record['source']}: "
                             f"{result.stderr[-4096:].decode(errors='replace')}")
        if len(result.stdout) != count * frame_bytes:
            continue
        check_source(record)
        values = np.frombuffer(result.stdout, dtype=np.uint8).reshape(count, frame_bytes)
        y = torch.from_numpy(values[:, :pixels].copy()).reshape(count, height, width)
        u = torch.from_numpy(values[:, pixels:pixels + pixels // 4].copy()).reshape(count, height // 2, width // 2)
        v = torch.from_numpy(values[:, pixels + pixels // 4:].copy()).reshape(count, height // 2, width // 2)
        return torch.stack((y, u.repeat_interleave(2, 1).repeat_interleave(2, 2),
                            v.repeat_interleave(2, 1).repeat_interleave(2, 2)), dim=1).float().div_(255)
    raise ValueError(f"video cannot supply {count} consecutive frames: {record['source']}")
