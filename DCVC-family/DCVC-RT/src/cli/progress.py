from __future__ import annotations

import json
import math
import os
import re
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, Optional, Set

from tqdm import tqdm


ENCODED_BIN_RE = re.compile(r"^(?P<base>.+)_\d+x\d+_qI\d+_qP\d+\.bin$")
ENCODED_BIN_STEM_RE = re.compile(r"^(?P<base>.+)_\d+x\d+_qI\d+_qP\d+$")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def fmt_hhmmss(total_seconds: float) -> str:
    total_seconds = max(0, int(total_seconds))
    hh = total_seconds // 3600
    mm = (total_seconds % 3600) // 60
    ss = total_seconds % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def progress_log_path(out_dir: Path, filename: str = ".progress.jsonl") -> Path:
    return out_dir / filename


def append_progress_log(out_dir: Path, record: Dict, filename: str = ".progress.jsonl"):
    ensure_dir(out_dir)
    rec = dict(record)
    rec.setdefault("at", now_iso())
    with progress_log_path(out_dir, filename).open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def parse_encode_resume_state(log_file: Path) -> Dict[str, Dict[str, str]]:
    state: Dict[str, Dict[str, str]] = {}
    if not log_file.exists():
        return state

    with log_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            vid_id = rec.get("video_id")
            if not vid_id:
                continue

            if vid_id not in state:
                state[vid_id] = {"video_status": "unknown", "audio_status": "unknown"}

            status = rec.get("status")
            if status == "video-done":
                state[vid_id]["video_status"] = "done"
            elif status == "started":
                state[vid_id]["video_status"] = "started"
                state[vid_id]["audio_status"] = "started"
            elif status == "failed":
                stage = rec.get("stage")
                if stage == "audio":
                    state[vid_id]["audio_status"] = "failed"
                elif stage == "video":
                    state[vid_id]["video_status"] = "failed"

    return state


def parse_decode_resume_state(log_file: Path) -> Dict[str, str]:
    state: Dict[str, str] = {}
    if not log_file.exists():
        return state

    with log_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            video_id = rec.get("video_id")
            if not video_id:
                continue
            status = rec.get("status")
            if status == "decode-done":
                state[video_id] = "done"
            elif status == "decode-started":
                state.setdefault(video_id, "started")
            elif status == "failed" and rec.get("stage") == "decode":
                state[video_id] = "failed"

    return state


def encoded_base_from_bin_name(name: str) -> Optional[str]:
    match = ENCODED_BIN_RE.match(name)
    if match is None:
        return None
    return match.group("base")


def original_base_from_encoded_stem(stem: str) -> str:
    match = ENCODED_BIN_STEM_RE.match(stem)
    if match is None:
        return stem
    return match.group("base")


def decoded_base_from_yuv_name(name: str) -> Optional[str]:
    if not name.lower().endswith(".yuv"):
        return None

    stem = Path(name).stem
    base, sep, tail = stem.rpartition("_")
    if sep and tail.endswith("kbps") and tail[:-4].isdigit():
        return base
    return stem


@dataclass
class EncodeOutputIndex:
    video_bases: Set[str]
    audio_bases: Set[str]
    log_state: Dict[str, Dict[str, str]]


@dataclass
class DecodeOutputIndex:
    decoded_bases: Set[str]
    log_state: Dict[str, str]


def build_encode_output_index(channel_out: Path) -> EncodeOutputIndex:
    video_bases: Set[str] = set()
    audio_bases: Set[str] = set()
    if channel_out.exists():
        with os.scandir(channel_out) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                if entry.name.endswith(".opus"):
                    audio_bases.add(Path(entry.name).stem)
                    continue
                base = encoded_base_from_bin_name(entry.name)
                if base is not None:
                    video_bases.add(base)

    return EncodeOutputIndex(
        video_bases=video_bases,
        audio_bases=audio_bases,
        log_state=parse_encode_resume_state(progress_log_path(channel_out)),
    )


def build_decode_output_index(output_dir: Path) -> DecodeOutputIndex:
    decoded_bases: Set[str] = set()
    if output_dir.exists():
        with os.scandir(output_dir) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                base = decoded_base_from_yuv_name(entry.name)
                if base is not None:
                    decoded_bases.add(base)

    return DecodeOutputIndex(
        decoded_bases=decoded_bases,
        log_state=parse_decode_resume_state(progress_log_path(output_dir)),
    )


class AverageCompletionTime:
    """Simple per-video average and parallel-worker ETA estimator."""

    def __init__(self, worker_count: int):
        self.worker_count = max(1, worker_count)
        self.completed_seconds = 0.0
        self.samples = 0

    def record(self, elapsed_seconds: Optional[float]):
        if elapsed_seconds is None:
            return
        try:
            elapsed_seconds = float(elapsed_seconds)
        except (TypeError, ValueError):
            return
        if elapsed_seconds <= 0 or not math.isfinite(elapsed_seconds):
            return
        self.completed_seconds += elapsed_seconds
        self.samples += 1

    @property
    def average_seconds(self) -> Optional[float]:
        if self.samples == 0:
            return None
        return self.completed_seconds / self.samples

    def eta_seconds(self, total: int, done: int) -> Optional[float]:
        average = self.average_seconds
        if average is None:
            return None
        remaining = max(0, total - done)
        return average * remaining / self.worker_count


class RollingFrameRate:
    """Frame completion rate over a bounded trailing time window."""

    def __init__(self, window_seconds: float = 3.0, start_time: Optional[float] = None):
        if window_seconds <= 0 or not math.isfinite(window_seconds):
            raise ValueError("rolling FPS window must be a positive finite number")
        self.window_seconds = float(window_seconds)
        self.start_time = time.monotonic() if start_time is None else float(start_time)
        self.frame_times: Deque[float] = deque()

    def record_frame(self, timestamp: Optional[float] = None) -> float:
        now = time.monotonic() if timestamp is None else float(timestamp)
        if now < self.start_time:
            raise ValueError("frame timestamp cannot precede the rate start time")
        if self.frame_times and now < self.frame_times[-1]:
            raise ValueError("frame timestamps must be monotonic")

        self.frame_times.append(now)
        cutoff = now - self.window_seconds
        while self.frame_times and self.frame_times[0] < cutoff:
            self.frame_times.popleft()

        observed_seconds = min(self.window_seconds, now - self.start_time)
        return len(self.frame_times) / max(1e-6, observed_seconds)


class MultiWorkerProgress:
    def __init__(self, total: int, worker_count: int, prefix: str):
        self.total = total
        self.done = 0
        self.prefix = prefix
        self.t0 = time.time()
        self.completion_time = AverageCompletionTime(worker_count)
        self.global_bar = tqdm(
            total=total,
            position=0,
            bar_format="{desc}",
            dynamic_ncols=True,
        )
        self.worker_bars = [
            tqdm(
                total=1,
                position=i + 1,
                bar_format="{desc}",
                dynamic_ncols=True,
            )
            for i in range(worker_count)
        ]
        for i in range(worker_count):
            self.clear_worker(i)
        self._refresh_global()

    def _refresh_global(self):
        elapsed = time.time() - self.t0
        pct = (100.0 * self.done / self.total) if self.total else 100.0
        average = self.completion_time.average_seconds
        eta = self.completion_time.eta_seconds(self.total, self.done)
        estimate = (
            f" avg/video={fmt_hhmmss(average)} eta={fmt_hhmmss(eta)}"
            if average is not None and eta is not None
            else " avg/video=--:--:-- eta=--:--:--"
        )
        self.global_bar.set_description_str(
            f"{self.prefix}: {pct:05.2f}% {self.done}/{self.total} "
            f"elapsed={fmt_hhmmss(elapsed)}{estimate}"
        )
        self.global_bar.refresh()

    def set_done(self, done: int):
        self.done = done
        self._refresh_global()

    def increment_done(self, step: int = 1, elapsed_seconds: Optional[float] = None):
        self.done += step
        self.completion_time.record(elapsed_seconds)
        self._refresh_global()

    def set_worker_text(self, wid: int, text: str):
        if 0 <= wid < len(self.worker_bars):
            self.worker_bars[wid].set_description_str(text)
            self.worker_bars[wid].refresh()

    def clear_worker(self, wid: int):
        self.set_worker_text(wid, f"[{wid}] idle")

    def close(self):
        self.global_bar.close()
        for bar in self.worker_bars:
            bar.close()
