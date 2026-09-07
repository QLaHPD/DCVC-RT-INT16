from __future__ import annotations

import argparse
import io
import json
import os
import queue
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from multiprocessing import get_context

import numpy as np
import torch

from src.cli.channel_lifecycle import write_channel_inventory
from src.cli.dashboard import EncodeDashboard
from src.cli.jetson_decode import DecodeError, start_jetson_pipe, validate_decode
from src.cli.device_plan import build_worker_device_plan
from src.cli.lifecycle_runtime import ChannelLifecycleController, ChannelRuntimeState
from src.cli.progress import (
    RollingFrameRate,
    append_progress_log,
    build_encode_output_index,
    ensure_dir,
)
from src.layers.cuda_inference import replicate_pad
from src.models.image_model import DMCI
from src.models.video_model import DMC
from src.utils.common import load_model_for_inference, set_torch_env
from src.utils.stream_helper import SPSHelper, write_ip, write_sps
from src.utils.transforms import ycbcr420_to_444_np


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ.setdefault("PYTHONWARNINGS", "ignore")


CHILD_PROCS: List[subprocess.Popen] = []
TMP_RAM_PATHS: List[Path] = []
STOP_FLAG = False


@dataclass
class EncoderCfg:
    qp_i: int = 32
    qp_p: int = 10
    force_intra_period: int = -1
    reset_interval: int = 32
    resolution: Optional[int] = 72
    force_zero_thres: Optional[float] = None
    pad_multiple: int = 16
    ff_color_matrix: str = "bt709"
    ff_hwaccel: str = "auto"
    fps: Optional[float] = None
    ffmpeg_prefetch: int = 8


@dataclass
class EncodeTask:
    channel_id: str
    video_path: str
    need_video: bool
    need_audio: bool


@dataclass
class ChannelEncodeDiscovery:
    channel_id: str
    pending_tasks: List[EncodeTask]
    source_paths: List[str]
    total_found: int
    already_done: int
    warning: Optional[str] = None


@dataclass
class RawYuv420Frame:
    y: bytes
    u: bytes
    v: bytes


_FRAME_QUEUE_EOF = object()


class EncodeInterrupted(RuntimeError):
    pass


def best_ram_dir(preferred: Optional[str] = None) -> Path:
    if preferred:
        preferred_path = Path(preferred).expanduser()
        if preferred_path.exists() and os.access(preferred_path, os.W_OK):
            return preferred_path
    shm = Path("/dev/shm")
    if shm.exists() and os.access(shm, os.W_OK):
        return shm
    return Path(tempfile.gettempdir())


def register_tmp_ram(path: Path):
    TMP_RAM_PATHS.append(path)


def safe_unlink(path: Path):
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def kill_popen(proc: Optional[subprocess.Popen]):
    try:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except Exception:
                proc.kill()
    except Exception:
        pass


def kill_all_children():
    for proc in CHILD_PROCS:
        kill_popen(proc)


def cleanup_registered_ram_tmp():
    for path in TMP_RAM_PATHS:
        safe_unlink(path)
    TMP_RAM_PATHS.clear()


def release_cuda():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _fsync_file(fobj):
    fobj.flush()
    os.fsync(fobj.fileno())


def atomic_copy_across_fs(src: Path, dst: Path):
    ensure_dir(dst.parent)
    tmp = dst.with_suffix(dst.suffix + f".tmp.{os.getpid()}")
    try:
        with open(src, "rb") as fi, open(tmp, "wb") as fo:
            shutil.copyfileobj(fi, fo, length=1024 * 1024)
            _fsync_file(fo)
        os.replace(tmp, dst)
    except BaseException:
        safe_unlink(tmp)
        raise


def write_bytes_final(dst: Path, data: bytes, mode: str):
    del mode  # final outputs are always committed atomically to avoid partial files
    ensure_dir(dst.parent)
    tmp = dst.with_suffix(dst.suffix + f".tmp.{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            _fsync_file(f)
        os.replace(tmp, dst)
    except BaseException:
        safe_unlink(tmp)
        raise


def run_command(cmd: List[str]) -> Tuple[int, str, str]:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    CHILD_PROCS.append(proc)
    out, err = proc.communicate()
    return proc.returncode, out, err


def popen_command(cmd: List[str], **kwargs) -> subprocess.Popen:
    proc = subprocess.Popen(cmd, **kwargs)
    CHILD_PROCS.append(proc)
    return proc


def _parse_rate(rate: Optional[str]) -> Optional[float]:
    if not rate:
        return None
    if "/" in rate:
        num, den = rate.split("/", 1)
        try:
            return float(num) / float(den)
        except Exception:
            return None
    try:
        return float(rate)
    except Exception:
        return None


def probe_stream_props(input_url: str) -> Tuple[int, int, Optional[float]]:
    for _ in range(3):
        try:
            rc, out, _ = run_command([
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate",
                "-of", "json", input_url,
            ])
            if rc != 0:
                return 0, 0, None
            data = json.loads(out or "{}")
            stream = data.get("streams", [{}])[0]
            width = int(stream.get("width", 0) or 0)
            height = int(stream.get("height", 0) or 0)
            fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate"))
            return width, height, fps
        except Exception:
            time.sleep(1)
    return 0, 0, None


def encode_audio_opus_from_file_to_temp(video_path: str, tmp_out_file: Path, bitrate: str, frame_ms: float,
                                        complexity: int, channels: int, vbr: str):
    ff = [
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-nostats", "-y",
        "-i", video_path,
        "-vn", "-c:a", "libopus",
        "-b:a", str(bitrate),
        "-vbr", vbr,
        "-compression_level", str(complexity),
        "-frame_duration", str(frame_ms),
        "-ac", str(channels),
        str(tmp_out_file),
    ]
    proc = popen_command(ff, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rc = proc.wait()
    if rc != 0:
        safe_unlink(tmp_out_file)
        raise RuntimeError(f"ffmpeg opus failed rc={rc}")


def build_ffmpeg_chain_local(path: str, width: int, height: int, cfg: EncoderCfg,
                             source_width: Optional[int] = None,
                             source_height: Optional[int] = None) -> List[str]:
    filters = []
    if source_width != width or source_height != height:
        filters.append(
            f"scale={width}:{height}:flags=fast_bilinear+full_chroma_int:"
            f"in_color_matrix={cfg.ff_color_matrix}:out_color_matrix={cfg.ff_color_matrix}"
        )
    filters.extend(("format=yuv420p", "setsar=1/1"))
    if cfg.fps and cfg.fps > 0:
        filters.append(f"fps={cfg.fps}:round=down")
    vf = ",".join(filters)
    return [
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-nostats",
        "-fflags", "+genpts", "-reinit_filter", "0",
        "-hwaccel", "none" if cfg.ff_hwaccel == "jetson" else cfg.ff_hwaccel,
        "-i", path,
        "-vf", vf,
        "-pix_fmt", "yuv420p",
        "-vsync", "0",
        "-threads", "0",
        "-f", "rawvideo", "-",
    ]


def round_to_multiple(value: int, multiple: int) -> int:
    return int(round(value / multiple)) * multiple if value % multiple else value


def compute_target_dims(src_w: int, src_h: int, resolution: Optional[int], multiple: int) -> Tuple[int, int]:
    if not resolution:
        width, height = src_w, src_h
    elif src_h < src_w:
        height = resolution
        width = int(round(resolution * src_w / src_h))
    elif src_w < src_h:
        width = resolution
        height = int(round(resolution * src_h / src_w))
    else:
        width = resolution
        height = resolution

    width = round_to_multiple((width // 2) * 2, multiple)
    height = round_to_multiple((height // 2) * 2, multiple)
    width = (width // 2) * 2
    height = (height // 2) * 2
    return width, height


def build_local_basename(video_path: Path) -> str:
    return video_path.stem


def iter_input_files(channel_dir: Path, recursive: bool, extensions: set[str]) -> List[Path]:
    files: List[Path] = []
    if recursive:
        for path in channel_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in extensions:
                files.append(path)
    else:
        with os.scandir(channel_dir) as it:
            for entry in it:
                if entry.is_file() and Path(entry.name).suffix.lower() in extensions:
                    files.append(Path(entry.path))
    files.sort(key=lambda p: str(p))
    return files


def discover_channel_tasks(channel_id: str, in_root: Path, out_root: Path, recursive: bool,
                           extensions: set[str], audio_enabled: bool) -> ChannelEncodeDiscovery:
    channel_dir = in_root / channel_id
    if not channel_dir.exists() or not channel_dir.is_dir():
        return ChannelEncodeDiscovery(
            channel_id=channel_id,
            pending_tasks=[],
            source_paths=[],
            total_found=0,
            already_done=0,
            warning=f"Warning: channel folder not found: {channel_id}",
        )

    files = iter_input_files(channel_dir, recursive, extensions)
    channel_out = out_root / channel_id
    ensure_dir(channel_out)
    output_index = build_encode_output_index(channel_out)

    pending_tasks: List[EncodeTask] = []
    already_done = 0
    for video_path in files:
        base = build_local_basename(video_path)
        video_done = base in output_index.video_bases and \
            output_index.log_state.get(base, {}).get("video_status") == "done"
        audio_done = (not audio_enabled) or (base in output_index.audio_bases)
        if video_done and audio_done:
            already_done += 1
            continue
        pending_tasks.append(EncodeTask(
            channel_id=channel_id,
            video_path=str(video_path),
            need_video=not video_done,
            need_audio=audio_enabled and not audio_done,
        ))

    return ChannelEncodeDiscovery(
        channel_id=channel_id,
        pending_tasks=pending_tasks,
        source_paths=[str(path) for path in files],
        total_found=len(files),
        already_done=already_done,
    )


class NeuralEncoder:
    def __init__(self, device: torch.device, model_i_path: str, model_p_path: str, cfg: EncoderCfg):
        set_torch_env()
        self.device = device
        self.cfg = cfg
        self.i_net = load_model_for_inference(DMCI(), model_i_path, device, cfg.force_zero_thres)
        self.p_net = load_model_for_inference(DMC(), model_p_path, device, cfg.force_zero_thres)

    @staticmethod
    def _black_frame_yuv420p(width: int, height: int) -> Tuple[bytes, bytes, bytes]:
        y = np.full((height, width), 16, dtype=np.uint8)
        u = np.full((height // 2, width // 2), 128, dtype=np.uint8)
        v = np.full((height // 2, width // 2), 128, dtype=np.uint8)
        return y.tobytes(), u.tobytes(), v.tobytes()

    @staticmethod
    def _read_yuv420_frame(ffmpeg_proc, width: int, height: int) -> Optional[RawYuv420Frame]:
        y_size = width * height
        uv_size = (width // 2) * (height // 2)

        y = ffmpeg_proc.stdout.read(y_size)
        if not y:
            return None
        if len(y) < y_size:
            raise DecodeError("truncated Y plane")

        u = ffmpeg_proc.stdout.read(uv_size)
        v = ffmpeg_proc.stdout.read(uv_size)
        if (not u or len(u) < uv_size) or (not v or len(v) < uv_size):
            raise DecodeError("truncated chroma plane")

        return RawYuv420Frame(y=y, u=u, v=v)

    def _start_ffmpeg_prefetch(self, ffmpeg_proc, width: int, height: int):
        frame_queue = queue.Queue(maxsize=max(1, int(self.cfg.ffmpeg_prefetch)))
        stop_event = threading.Event()

        def _reader():
            try:
                while not STOP_FLAG and not stop_event.is_set():
                    frame = self._read_yuv420_frame(ffmpeg_proc, width, height)
                    if frame is None:
                        break
                    while not STOP_FLAG and not stop_event.is_set():
                        try:
                            frame_queue.put(frame, timeout=0.1)
                            break
                        except queue.Full:
                            continue
            except Exception as exc:  # pylint: disable=W0718
                while not STOP_FLAG and not stop_event.is_set():
                    try:
                        frame_queue.put(exc, timeout=0.1)
                        break
                    except queue.Full:
                        continue
            finally:
                while not stop_event.is_set():
                    try:
                        frame_queue.put(_FRAME_QUEUE_EOF, timeout=0.1)
                        break
                    except queue.Full:
                        continue

        reader_thread = threading.Thread(target=_reader, name=f"ffmpeg-prefetch-{os.getpid()}", daemon=True)
        reader_thread.start()
        return frame_queue, stop_event, reader_thread

    @torch.no_grad()
    def encode_from_ffmpeg_rawpipe(self, ffmpeg_proc, width: int, height: int, out_path: Path,
                                   log_dir: Path, video_id: str, on_progress=None, finalize_mode: str = "direct"):
        padding_r, padding_b = DMCI.get_padding_size(height, width, self.cfg.pad_multiple)
        use_two_ec = (width * height) > (1280 * 720)
        self.i_net.set_use_two_entropy_coders(use_two_ec)
        self.p_net.set_use_two_entropy_coders(use_two_ec)
        self.p_net.set_curr_poc(0)

        output_buff = io.BytesIO()
        sps_helper = SPSHelper()
        frame_idx = 0
        last_qp = 0
        append_progress_log(log_dir, {
            "video_id": video_id,
            "stage": "debug",
            "event": "start_seq",
            "w": width,
            "h": height,
            "pad_b": padding_b,
            "pad_r": padding_r,
        })

        t0 = time.monotonic()
        t_last = t0
        rolling_fps = RollingFrameRate(window_seconds=3.0, start_time=t0)
        frame_queue, prefetch_stop, reader_thread = self._start_ffmpeg_prefetch(ffmpeg_proc, width, height)

        try:
            while not STOP_FLAG:
                try:
                    frame = frame_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if STOP_FLAG:
                    raise EncodeInterrupted("encode interrupted")
                if frame is _FRAME_QUEUE_EOF:
                    break
                if isinstance(frame, Exception):
                    raise frame

                y_np = np.frombuffer(frame.y, dtype=np.uint8).reshape(1, height, width)
                uv_np = np.stack([
                    np.frombuffer(frame.u, dtype=np.uint8).reshape(height // 2, width // 2),
                    np.frombuffer(frame.v, dtype=np.uint8).reshape(height // 2, width // 2),
                ])
                x444 = ycbcr420_to_444_np(y_np, uv_np)
                x = torch.from_numpy(x444).to(self.device, dtype=torch.float32) / 255.0
                x = x.unsqueeze(0).half()
                x_padded = replicate_pad(x, padding_b, padding_r).contiguous()

                is_i = frame_idx == 0 or (
                    self.cfg.force_intra_period > 0 and frame_idx % self.cfg.force_intra_period == 0
                )
                with torch.cuda.amp.autocast(enabled=False):
                    if is_i:
                        curr_qp = self.cfg.qp_i
                        use_ada_i = 0
                        encoded = self.i_net.compress(x_padded, curr_qp)
                        self.p_net.clear_dpb()
                        self.p_net.add_ref_frame(None, encoded["x_hat"])
                    else:
                        use_ada_i = 1 if (
                            self.cfg.reset_interval > 0 and frame_idx % self.cfg.reset_interval == 1
                        ) else 0
                        if use_ada_i:
                            self.p_net.prepare_feature_adaptor_i(last_qp)
                        curr_qp = self.p_net.shift_qp(
                            self.cfg.qp_p, [0, 1, 0, 2, 0, 2, 0, 2][frame_idx % 8]
                        )
                        encoded = self.p_net.compress(x_padded, curr_qp)
                        last_qp = curr_qp

                sps = {
                    "height": height,
                    "width": width,
                    "ec_part": int(use_two_ec),
                    "use_ada_i": use_ada_i,
                }
                sps_id, is_new_sps = sps_helper.get_sps_id(sps)
                sps["sps_id"] = sps_id
                if is_new_sps:
                    write_sps(output_buff, sps)
                write_ip(output_buff, is_i, sps_id, curr_qp, encoded["bit_stream"])

                frame_idx += 1
                now = time.monotonic()
                current_fps = rolling_fps.record_frame(now)
                if on_progress and (now - t_last) >= 0.1:
                    elapsed = now - t0
                    on_progress(frame_idx, current_fps, elapsed)
                    t_last = now
        finally:
            prefetch_stop.set()
            try:
                reader_thread.join(timeout=1.0)
            except Exception:
                pass

        if STOP_FLAG:
            raise EncodeInterrupted("encode interrupted")

        validate_decode(ffmpeg_proc, frame_idx)

        append_progress_log(log_dir, {
            "video_id": video_id,
            "stage": "stats",
            "event": "frames_encoded",
            "frames": frame_idx,
        })
        write_bytes_final(out_path, output_buff.getbuffer(), mode=finalize_mode)


def process_one_file(task: EncodeTask, progress_q, wid: int, channel_out: Path, encoder: NeuralEncoder,
                     enc_cfg: EncoderCfg, audio_enable: bool, opus_params: Dict, finalize_mode: str) -> Tuple[bool, Optional[float]]:
    video_path = Path(task.video_path)
    base = build_local_basename(video_path)
    ensure_dir(channel_out)

    def emit(event_type: str, **fields):
        progress_q.put({
            "type": event_type,
            "wid": wid,
            "vid": base,
            "channel_id": task.channel_id,
            "source_path": str(video_path),
            **fields,
        })

    emit("worker_task_start")

    if not task.need_video and not task.need_audio:
        emit("worker_done", frames=0, elapsed=0.0)
        return True, 0.0

    append_progress_log(channel_out, {"video_id": base, "status": "started", "file": str(video_path)})

    audio_thread = None
    audio_result = {"ok": False, "error": None}
    opus_path = channel_out / f"{base}.opus"
    if task.need_audio and audio_enable:
        tmp_dir = Path(os.environ.get("ENC_RAM_TMP_DIR", best_ram_dir()))
        tmp_opus = tmp_dir / f".tmp_{base}_{os.getpid()}.opus"
        register_tmp_ram(tmp_opus)
        emit("worker_audio", status="started")

        def _audio_worker():
            try:
                encode_audio_opus_from_file_to_temp(
                    str(video_path),
                    tmp_opus,
                    bitrate=opus_params["bitrate"],
                    frame_ms=opus_params["frame_ms"],
                    complexity=opus_params["complexity"],
                    channels=opus_params["channels"],
                    vbr=opus_params["vbr"],
                )
                audio_result["ok"] = True
            except Exception as exc:
                audio_result["error"] = str(exc)
                emit("worker_audio", status="failed")
                append_progress_log(channel_out, {
                    "video_id": base,
                    "status": "failed",
                    "stage": "audio",
                    "error": str(exc),
                })

        audio_thread = threading.Thread(target=_audio_worker, name=f"aud-{base}", daemon=True)
        audio_thread.start()
    else:
        emit("worker_audio", status="done" if audio_enable and not task.need_audio else "off")

    if not task.need_video:
        if audio_thread:
            audio_thread.join()
        if task.need_audio and audio_enable:
            if STOP_FLAG:
                safe_unlink(tmp_opus)
                raise EncodeInterrupted("encode interrupted")
            if audio_result["ok"]:
                atomic_copy_across_fs(tmp_opus, opus_path)
                safe_unlink(tmp_opus)
                emit("worker_audio", status="done")
                emit("worker_done", frames=0, elapsed=0.0)
                return True, 0.0
            safe_unlink(tmp_opus)
            emit("worker_fail", stage="audio", error=audio_result["error"] or "audio encode failed")
            return False, None
        emit("worker_done", frames=0, elapsed=0.0)
        return True, 0.0

    t0 = time.time()
    ff_proc = None
    try:
        src_w, src_h, src_fps = probe_stream_props(str(video_path))
        if src_w <= 0 or src_h <= 0:
            src_w, src_h = 854, 480
        new_w, new_h = compute_target_dims(src_w, src_h, enc_cfg.resolution, enc_cfg.pad_multiple)
        ff_cmd = build_ffmpeg_chain_local(
            str(video_path),
            new_w,
            new_h,
            enc_cfg,
            source_width=src_w,
            source_height=src_h,
        )
        resolution_text = f"{src_w}, {src_h} -> {new_w}, {new_h}"

        bin_path = channel_out / f"{base}_{new_w}x{new_h}_qI{enc_cfg.qp_i}_qP{enc_cfg.qp_p}.bin"
        append_progress_log(channel_out, {
            "video_id": base,
            "status": "video-encoding",
            "ff_cmd": ff_cmd,
            "src_fps": src_fps,
            "out_fps": enc_cfg.fps or "source",
            "source_width": src_w,
            "source_height": src_h,
            "output_width": new_w,
            "output_height": new_h,
            "resize_applied": (src_w, src_h) != (new_w, new_h),
        })
        emit(
            "worker_start",
            audio="on" if task.need_audio and audio_enable else ("done" if audio_enable else "off"),
            resolution=resolution_text,
        )

        def on_progress(frames, fps, elapsed):
            emit("worker_prog", frames=int(frames), fps=float(fps), elapsed=float(elapsed))

        modes = ["jetson", "none"] if enc_cfg.ff_hwaccel == "jetson" else [enc_cfg.ff_hwaccel]
        for mode in modes:
            try:
                ff_proc = (start_jetson_pipe(str(video_path), ff_cmd, popen_command)
                           if mode == "jetson" else
                           popen_command(ff_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL))
                append_progress_log(channel_out, {"video_id": base, "event": "decoder", "backend": mode})
                encoder.encode_from_ffmpeg_rawpipe(
                    ff_proc, new_w, new_h, bin_path, log_dir=channel_out,
                    video_id=base, on_progress=on_progress, finalize_mode=finalize_mode,
                )
                break
            except DecodeError as exc:
                if mode != "jetson" or STOP_FLAG:
                    raise
                append_progress_log(channel_out, {
                    "video_id": base, "event": "decoder_fallback", "error": str(exc),
                    "backend": "none",
                })
            finally:
                if ff_proc is not None:
                    if hasattr(ff_proc, "close"):
                        ff_proc.close()
                    else:
                        kill_popen(ff_proc)
                        ff_proc.stdout.close()
                    ff_proc = None

        append_progress_log(channel_out, {"video_id": base, "status": "video-done", "out_bin": str(bin_path)})

        if audio_thread:
            audio_thread.join()
            if STOP_FLAG:
                safe_unlink(tmp_opus)
                raise EncodeInterrupted("encode interrupted")
            if audio_result["ok"]:
                atomic_copy_across_fs(tmp_opus, opus_path)
                emit("worker_audio", status="done")
            else:
                safe_unlink(tmp_opus)
                emit("worker_fail", stage="audio", error=audio_result["error"] or "audio encode failed")
                return False, None

        dur = time.time() - t0
        emit("worker_done", frames=None, elapsed=dur)
        return True, dur
    except Exception as exc:
        append_progress_log(channel_out, {
            "video_id": base,
            "status": "failed",
            "stage": "video",
            "error": str(exc),
        })
        emit("worker_fail", stage="video", error=str(exc))
        return False, None
    finally:
        kill_popen(ff_proc)
        if audio_thread:
            try:
                audio_thread.join(timeout=1.0)
            except Exception:
                pass
        if task.need_audio and audio_enable:
            safe_unlink(tmp_opus)


def _worker_sig_handler(sig, frame):
    del sig, frame
    global STOP_FLAG
    STOP_FLAG = True
    kill_all_children()
    cleanup_registered_ram_tmp()
    release_cuda()
    raise SystemExit(130)


def worker_entry(wid: int, task_q, progress_q, stop_event, output_root: str, model_i: str, model_p: str,
                 enc_cfg_dict: Dict, audio_mode: str, opus_params: Dict, use_cuda: bool,
                 cuda_device_index: Optional[int], finalize_mode: str):
    global STOP_FLAG
    STOP_FLAG = False
    signal.signal(signal.SIGTERM, _worker_sig_handler)
    signal.signal(signal.SIGINT, _worker_sig_handler)

    set_torch_env()
    if use_cuda and torch.cuda.is_available():
        if cuda_device_index is None:
            cuda_device_index = 0
        torch.cuda.set_device(cuda_device_index)
        device = torch.device(f"cuda:{cuda_device_index}")
    else:
        device = torch.device("cpu")

    encoder = NeuralEncoder(device, model_i, model_p, EncoderCfg(**enc_cfg_dict))
    progress_q.put({
        "type": "worker_hello",
        "wid": wid,
        "pid": os.getpid(),
        "device": str(device),
    })

    while not stop_event.is_set():
        try:
            task = task_q.get(timeout=0.5)
        except queue.Empty:
            continue
        if task is None:
            break

        channel_out = Path(output_root) / task.channel_id
        process_one_file(
            task=task,
            progress_q=progress_q,
            wid=wid,
            channel_out=channel_out,
            encoder=encoder,
            enc_cfg=EncoderCfg(**enc_cfg_dict),
            audio_enable=(audio_mode == "opus"),
            opus_params=opus_params,
            finalize_mode=finalize_mode,
        )

    kill_all_children()
    cleanup_registered_ram_tmp()
    release_cuda()


def configure_parser(parser: argparse.ArgumentParser):
    io_group = parser.add_argument_group("Input/Output Paths")
    io_group.add_argument("--base_root", required=True, help="Root directory containing per-channel input folders.")
    io_group.add_argument("--output_root", required=True, help="Root directory where encoded output folders will be saved.")
    io_group.add_argument("--channel_ids", nargs="+", default=None, help="Space-separated list of channel folder names to process.")
    io_group.add_argument("--twitch_channels", nargs="+", default=None, help="Space-separated list of Twitch channel folder names.")

    hw_group = parser.add_argument_group("Hardware and Concurrency")
    hw_group.add_argument(
        "--procs",
        type=int,
        default=None,
        help=(
            "Number of independent video workers. By default, CUDA uses one worker per "
            "selected/visible GPU and CPU mode uses one worker."
        ),
    )
    hw_group.add_argument("--cuda", type=lambda s: s.lower() in {"1", "true", "yes"}, default=True, help="Enable or disable CUDA acceleration.")
    hw_group.add_argument(
        "--cuda_idx",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Logical CUDA device indices to use. Defaults to every visible GPU; workers are "
            "assigned round-robin when --procs exceeds the selected GPU count."
        ),
    )

    vid_group = parser.add_argument_group("Neural Video Encoder")
    vid_group.add_argument("--model_path_i", default="./checkpoints/cvpr2025_image.pth.tar", help="Path to the neural image model checkpoint.")
    vid_group.add_argument("--model_path_p", default="./checkpoints/cvpr2025_video.pth.tar", help="Path to the neural video model checkpoint.")
    vid_group.add_argument("--qp_i", type=int, default=32, help="I-frame quantization parameter.")
    vid_group.add_argument("--qp_p", type=int, default=10, help="P-frame quantization parameter.")
    vid_group.add_argument("--resolution", type=int, default=96, help="Target scaling resolution (short edge).")
    vid_group.add_argument("--pad_multiple", type=int, default=16, help="Padding multiple for image dimensions.")
    vid_group.add_argument("--force_intra_period", type=int, default=-1, help="Force an I-frame every N frames.")
    vid_group.add_argument("--reset_interval", type=int, default=32, help="Interval for resetting the feature adapter.")
    vid_group.add_argument("--force_zero_thres", type=float, default=None, help="Optional threshold to force small latent values to zero.")

    aud_group = parser.add_argument_group("Audio (Opus)")
    aud_group.add_argument("--audio", choices=["opus", "none"], default="opus", help="Audio processing mode.")
    aud_group.add_argument("--opus_bitrate", default="12k", help="Target bitrate for Opus audio encoding.")
    aud_group.add_argument("--opus_frame_ms", type=float, default=60.0, help="Opus frame duration in milliseconds.")
    aud_group.add_argument("--opus_complexity", type=int, default=10, help="Opus encoder computational complexity [0-10].")
    aud_group.add_argument("--opus_channels", choices=["mono", "stereo"], default="stereo", help="Number of audio channels.")
    aud_group.add_argument("--opus_vbr", choices=["on", "off", "constrained"], default="on", help="Opus variable bitrate mode.")

    pipe_group = parser.add_argument_group("Pipeline and FFMPEG Settings")
    pipe_group.add_argument("--fps", type=float, default=30, help="Target framerate for the video.")
    pipe_group.add_argument("--ff_color_matrix", choices=["bt709", "bt601", "bt2020"], default="bt709", help="Color matrix used by FFmpeg during YUV conversion.")
    pipe_group.add_argument("--ff_hwaccel", choices=["none", "auto", "jetson"], default="none", help="Decoder: none/auto (FFmpeg), or jetson (GStreamer NVDEC for supported H.264 MP4/MOV; software fallback).")
    pipe_group.add_argument("--ffmpeg_prefetch", type=int, default=8, help="How many decoded raw frames FFmpeg may buffer ahead of the GPU loop.")
    pipe_group.add_argument("--extensions", nargs="+", default=[".mp4", ".mkv", ".avi", ".mov", ".webm"], help="File extensions to process.")
    pipe_group.add_argument("--recursive", type=lambda s: s.lower() in {"1", "true", "yes"}, default=False, help="Whether to search recursively within the channel folders.")
    pipe_group.add_argument("--ram_tmp_dir", default=None, help="Override path for the RAM temporary directory.")
    pipe_group.add_argument("--disk_finalize", choices=["direct", "atomic"], default="atomic", help="Compatibility option. Final outputs are committed atomically to avoid partial files.")

    lifecycle_group = parser.add_argument_group("Channel Lifecycle and Cleanup")
    lifecycle_group.add_argument(
        "--auto-delete",
        action="store_true",
        help="After strict channel validation, delete only the exact discovered source-video paths without prompting.",
    )
    lifecycle_group.add_argument(
        "--keep-originals",
        action="store_true",
        help="Do not offer or perform source-video cleanup. By default completed channels await TUI approval.",
    )
    lifecycle_group.add_argument(
        "--cleanup-dry-run",
        action="store_true",
        help="Run validation and write archive manifests/audit records, but do not unlink source videos.",
    )
    lifecycle_group.add_argument(
        "--allow-missing-metadata",
        action="store_true",
        help="Allow cleanup when a source has no matching .info.json sidecar.",
    )
    lifecycle_group.add_argument(
        "--ui",
        choices=["auto", "tui", "plain"],
        default="auto",
        help="Terminal interface. Auto uses the switchable TUI on a real terminal and plain output otherwise.",
    )


def _format_worker_text(state: Dict) -> str:
    vid = state.get("vid", "-")
    if len(vid) > 30:
        vid = vid[:27] + "..."
    device = state.get("device", "")
    device_text = f"{device} " if device else ""
    resolution = state.get("resolution", "")
    resolution_text = f" {resolution}" if resolution else ""
    return (
        f"{device_text}{vid}{resolution_text} {state.get('frames', 0)}f "
        f"{state.get('fps', 0.0):.1f}fps Aud:{state.get('audio', 'off')}"
    )


def _close_queue(mp_queue):
    try:
        mp_queue.cancel_join_thread()
    except Exception:
        pass
    try:
        mp_queue.close()
    except Exception:
        pass


def _terminate_workers(workers, grace_seconds: float = 3.0):
    for proc in workers:
        if proc.is_alive():
            proc.terminate()

    deadline = time.time() + grace_seconds
    for proc in workers:
        remaining = max(0.0, deadline - time.time())
        proc.join(timeout=remaining)

    for proc in workers:
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=1.0)


def run(args) -> int:
    if args.auto_delete and args.keep_originals:
        print("Error: --auto-delete and --keep-originals are mutually exclusive.")
        return 2

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

    ram_dir = best_ram_dir(args.ram_tmp_dir)
    os.environ["ENC_RAM_TMP_DIR"] = str(ram_dir)

    in_root = Path(args.base_root).resolve()
    out_root = Path(args.output_root)
    ensure_dir(out_root)
    out_root = out_root.resolve()

    sources: List[str] = []
    if args.channel_ids:
        sources.extend(args.channel_ids)
    if args.twitch_channels:
        sources.extend(args.twitch_channels)
    if not sources:
        print("Error: Provide --channel_ids and/or --twitch_channels.")
        return 2

    extensions = {ext.lower() for ext in args.extensions}
    audio_enabled = args.audio == "opus"
    max_scan_workers = min(len(sources), max(1, os.cpu_count() or 1, 4))
    discoveries: List[ChannelEncodeDiscovery] = []
    with ThreadPoolExecutor(max_workers=max_scan_workers) as executor:
        futures = [
            executor.submit(
                discover_channel_tasks,
                channel_id,
                in_root,
                out_root,
                args.recursive,
                extensions,
                audio_enabled,
            )
            for channel_id in sources
        ]
        for future in futures:
            discoveries.append(future.result())

    total_found = 0
    already_done = 0
    all_tasks: List[EncodeTask] = []
    for discovery in discoveries:
        if discovery.warning:
            print(discovery.warning)
        total_found += discovery.total_found
        already_done += discovery.already_done
        all_tasks.extend(discovery.pending_tasks)

    if total_found == 0:
        print("Nothing to do — no videos found.")
        return 0

    if args.cuda and not device_plan.using_cuda:
        print("Warning: --cuda specified but no devices found. Running on CPU.")

    enc_cfg_dict = dict(
        qp_i=args.qp_i,
        qp_p=args.qp_p,
        force_intra_period=args.force_intra_period,
        reset_interval=args.reset_interval,
        resolution=args.resolution,
        pad_multiple=args.pad_multiple,
        ff_color_matrix=args.ff_color_matrix,
        ff_hwaccel=args.ff_hwaccel,
        ffmpeg_prefetch=args.ffmpeg_prefetch,
        fps=args.fps,
        force_zero_thres=args.force_zero_thres,
    )
    opus_params = {
        "bitrate": args.opus_bitrate,
        "frame_ms": args.opus_frame_ms,
        "complexity": args.opus_complexity,
        "channels": 1 if args.opus_channels == "mono" else 2,
        "vbr": args.opus_vbr,
    }

    cleanup_policy = "auto" if args.auto_delete else ("keep" if args.keep_originals else "prompt")
    inventory_config = {
        **enc_cfg_dict,
        "audio": args.audio,
        "opus": dict(opus_params),
        "extensions": sorted(extensions),
    }
    runtime_channels: List[ChannelRuntimeState] = []
    for discovery in discoveries:
        if discovery.total_found <= 0:
            continue
        state_path = write_channel_inventory(
            channel_id=discovery.channel_id,
            input_dir=in_root / discovery.channel_id,
            output_dir=out_root / discovery.channel_id,
            source_paths=[Path(value) for value in discovery.source_paths],
            source_extensions=extensions,
            audio_required=audio_enabled,
            metadata_required=not args.allow_missing_metadata,
            encoder_config=inventory_config,
        )
        runtime_channels.append(ChannelRuntimeState(
            channel_id=discovery.channel_id,
            state_path=state_path,
            total=discovery.total_found,
            already_done=discovery.already_done,
            pending=len(discovery.pending_tasks),
        ))
    lifecycle = ChannelLifecycleController(
        runtime_channels,
        policy=cleanup_policy,
        dry_run=args.cleanup_dry_run,
    )

    ctx = get_context("spawn")
    task_q = ctx.Queue()
    progress_q = ctx.Queue()
    stop_event = ctx.Event()

    workers = []
    worker_count = min(device_plan.worker_count, len(all_tasks)) if all_tasks else 0
    cuda_plan = device_plan.cuda_indices[:worker_count]
    if device_plan.using_cuda:
        workers_description = ", ".join(
            f"worker {wid}->cuda:{cuda_plan[wid]}" for wid in range(worker_count)
        )
    elif worker_count:
        noun = "worker" if worker_count == 1 else "workers"
        workers_description = f"{worker_count} CPU {noun}"
    else:
        workers_description = "no workers needed"
    print(f"Worker plan: {workers_description}")
    for wid in range(worker_count):
        proc = ctx.Process(
            target=worker_entry,
            args=(
                wid,
                task_q,
                progress_q,
                stop_event,
                str(out_root),
                args.model_path_i,
                args.model_path_p,
                enc_cfg_dict,
                args.audio,
                opus_params,
                device_plan.using_cuda,
                cuda_plan[wid] if wid < len(cuda_plan) else None,
                args.disk_finalize,
            ),
            daemon=True,
        )
        proc.start()
        workers.append(proc)

    for task in all_tasks:
        task_q.put(task)
    for _ in workers:
        task_q.put(None)

    progress = EncodeDashboard(total_found, len(workers), out_root, mode=args.ui)
    progress.set_done(already_done)
    active: Dict[int, Dict] = {}
    expected_tasks = {
        (task.channel_id, build_local_basename(Path(task.video_path)))
        for task in all_tasks
    }
    terminal_tasks = set()
    workers_dead_since: Optional[float] = None
    unreported_worker_failure = False
    had_task_failure = False
    interrupted = False
    exit_approvals = False

    try:
        alive = bool(workers)
        while True:
            try:
                msg = progress_q.get(timeout=0.1)
            except queue.Empty:
                msg = None

            if msg:
                typ = msg.get("type")
                wid = msg.get("wid", 0)
                channel_id = msg.get("channel_id", "")
                video_id = msg.get("vid", "")
                if typ == "worker_hello":
                    progress.set_worker_text(
                        wid,
                        f"{msg.get('device', 'unknown')} ready (pid {msg.get('pid', '?')})",
                    )
                elif typ == "worker_task_start":
                    lifecycle.mark_started(channel_id, video_id)
                    active[wid] = {
                        "vid": video_id,
                        "device": (
                            f"cuda:{cuda_plan[wid]}"
                            if wid < len(cuda_plan) and cuda_plan[wid] is not None
                            else "cpu"
                        ),
                        "frames": 0,
                        "fps": 0.0,
                        "audio": "starting",
                    }
                    progress.set_worker_text(wid, _format_worker_text(active[wid]))
                elif typ == "worker_start":
                    active[wid] = {
                        "vid": video_id,
                        "device": (
                            f"cuda:{cuda_plan[wid]}"
                            if wid < len(cuda_plan) and cuda_plan[wid] is not None
                            else "cpu"
                        ),
                        "frames": 0,
                        "fps": 0.0,
                        "audio": msg.get("audio", "off"),
                        "resolution": msg.get("resolution", ""),
                    }
                    progress.set_worker_text(wid, _format_worker_text(active[wid]))
                elif typ == "worker_prog":
                    state = active.setdefault(wid, {"vid": video_id, "audio": "off"})
                    state.update({"frames": msg.get("frames", 0), "fps": msg.get("fps", 0.0)})
                    progress.set_worker_text(wid, _format_worker_text(state))
                elif typ == "worker_audio":
                    state = active.setdefault(wid, {"vid": video_id, "frames": 0, "fps": 0.0})
                    state["audio"] = msg.get("status", "off")
                    progress.set_worker_text(wid, _format_worker_text(state))
                elif typ in {"worker_done", "worker_fail"}:
                    task_key = (channel_id, video_id)
                    if task_key in terminal_tasks:
                        continue
                    terminal_tasks.add(task_key)
                    active.pop(wid, None)
                    progress.clear_worker(wid)
                    progress.increment_done(
                        elapsed_seconds=msg.get("elapsed") if typ == "worker_done" else None
                    )
                    had_task_failure = had_task_failure or typ == "worker_fail"
                    lifecycle.mark_terminal(
                        channel_id,
                        video_id,
                        success=(typ == "worker_done"),
                        error=str(msg.get("error", "")),
                    )

            alive = any(proc.is_alive() for proc in workers)
            if alive:
                workers_dead_since = None
            elif terminal_tasks != expected_tasks:
                if workers_dead_since is None:
                    workers_dead_since = time.monotonic()
                elif time.monotonic() - workers_dead_since >= 1.0:
                    missing = sorted(expected_tasks - terminal_tasks)
                    for channel_id, video_id in missing:
                        error = "encoder workers exited without reporting task completion"
                        lifecycle.mark_terminal(channel_id, video_id, success=False, error=error)
                        terminal_tasks.add((channel_id, video_id))
                        progress.increment_done()
                        progress.write(f"{channel_id}/{video_id}: {error}; originals retained")
                    if missing:
                        unreported_worker_failure = True
                        had_task_failure = True
            lifecycle.poll()
            progress.update_lifecycle(lifecycle.snapshots(), lifecycle.events())
            for action in progress.poll_actions():
                if action.kind == "approve" and action.channel_id:
                    lifecycle.approve(action.channel_id)
                elif action.kind == "keep" and action.channel_id:
                    lifecycle.keep(action.channel_id)
                elif action.kind == "retry" and action.channel_id:
                    lifecycle.retry(action.channel_id)
                elif action.kind == "exit-approvals":
                    exit_approvals = True

            queue_drained = progress_q.empty()
            if alive or not queue_drained or lifecycle.has_background_work():
                continue
            if (
                cleanup_policy == "prompt"
                and progress.interactive
                and lifecycle.awaiting_approvals()
                and not exit_approvals
            ):
                continue
            break
    except KeyboardInterrupt:
        interrupted = True
        stop_event.set()
        progress.global_bar.write("Interrupted. Stopping workers and discarding in-progress outputs...")
        _terminate_workers(workers)
    finally:
        progress.close()
        stop_event.set()
        if interrupted:
            _terminate_workers(workers, grace_seconds=1.0)
        else:
            for proc in workers:
                proc.join()
        _close_queue(task_q)
        _close_queue(progress_q)
        lifecycle.close(wait=not interrupted)
        kill_all_children()
        cleanup_registered_ram_tmp()

    pending_approvals = lifecycle.awaiting_approvals()
    if pending_approvals:
        print(
            "Cleanup approval remains pending for: " + ", ".join(pending_approvals) +
            ". Re-run the cleanup command later; no originals were deleted for these channels."
        )
    elif not all_tasks:
        print("All videos were already encoded; channel lifecycle checks completed.")

    lifecycle_failure = any(
        channel.failures or channel.status.endswith("-failed")
        for channel in lifecycle.channels.values()
    )
    if interrupted:
        return 130
    return 1 if had_task_failure or unreported_worker_failure or lifecycle_failure else 0
