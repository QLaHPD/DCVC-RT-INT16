from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import subprocess
import time
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Optional

import torch

from src.codec.frame_decoder import BitstreamFrameSource, DecoderModels, load_decoder_models, resolve_decode_device
from src.cli.device_plan import build_worker_device_plan
from src.cli.progress import (
    MultiWorkerProgress,
    RollingFrameRate,
    append_progress_log,
    build_decode_output_index,
    ensure_dir,
    original_base_from_encoded_stem,
)
from src.utils.common import str2bool
from src.utils.video_writer import YUV420Writer


STOP_FLAG = False


@dataclass
class DecodeTask:
    bin_path: str
    base_name: str
    original_path: Optional[str]


def extract_video_data(video_path: str):
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,nb_frames", "-of", "json", video_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)["streams"][0]
    frame_rate = eval(data.get("r_frame_rate", "30/1"))
    frame_count = int(data.get("nb_frames", 0))
    if frame_count == 0:
        count_command = [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames",
            "-of", "default=nokey=1:noprint_wrappers=1", video_path,
        ]
        count_result = subprocess.run(count_command, capture_output=True, text=True, check=True)
        frame_count = int(count_result.stdout.strip())
    return frame_rate, frame_count


def _worker_sig_handler(sig, frame):
    del sig, frame
    global STOP_FLAG
    STOP_FLAG = True
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def build_decode_tasks(args) -> tuple[List[DecodeTask], int, int]:
    input_dir = Path(args.input_folder)
    output_dir = Path(args.output_folder)
    ensure_dir(output_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Warning: Input folder '{args.input_folder}' not found.")
        return [], 0, 0

    bin_paths = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".bin")
    output_index = build_decode_output_index(output_dir)

    original_videos: Dict[str, str] = {}
    if args.original_folder:
        original_dir = Path(args.original_folder)
        if not original_dir.exists():
            print(f"Warning: Original folder '{args.original_folder}' not found. Continuing without it.")
            args.original_folder = None
        else:
            original_videos = {
                path.stem.lower(): str(path)
                for path in original_dir.iterdir()
                if path.is_file()
            }

    pending_tasks: List[DecodeTask] = []
    already_done = 0
    for bin_path in bin_paths:
        base_name = bin_path.stem
        if output_index.log_state.get(base_name) == "done" or base_name in output_index.decoded_bases:
            already_done += 1
            continue

        original_path = None
        if args.original_folder:
            original_base = original_base_from_encoded_stem(base_name).lower()
            original_path = original_videos.get(original_base)

        pending_tasks.append(DecodeTask(
            bin_path=str(bin_path),
            base_name=base_name,
            original_path=original_path,
        ))

    return pending_tasks, len(bin_paths), already_done


def run_decoding(task: DecodeTask, output_folder: str, progress_q, wid: int, models: DecoderModels):
    output_dir = Path(output_folder)
    append_progress_log(output_dir, {
        "video_id": task.base_name,
        "status": "decode-started",
        "file": task.bin_path,
    })

    source = BitstreamFrameSource(task.bin_path, models)
    frame_total = source.index.frame_count
    progress_q.put({
        "type": "worker_start",
        "wid": wid,
        "vid": task.base_name,
        "frame_total": frame_total,
    })
    bin_size = Path(task.bin_path).stat().st_size
    pic_height, pic_width = source.index.height, source.index.width

    total_kbps = 0
    if task.original_path and os.path.exists(task.original_path):
        try:
            frame_rate, original_frame_num = extract_video_data(task.original_path)
            if original_frame_num > 0 and frame_rate > 0:
                total_kbps = int(bin_size * 8 / (original_frame_num / frame_rate) / 1000)
        except (subprocess.CalledProcessError, json.JSONDecodeError, IndexError, ValueError):
            total_kbps = 0

    output_name = (
        f"{task.base_name}_{total_kbps}kbps.yuv"
        if total_kbps > 0
        else f"{task.base_name}.yuv"
    )
    output_path = output_dir / output_name
    recon_writer = YUV420Writer(str(output_path), pic_width, pic_height)

    frame_idx = 0
    t0 = time.monotonic()
    t_last = t0
    rolling_fps = RollingFrameRate(window_seconds=3.0, start_time=t0)

    try:
        for frame in source.iter_frames(output_format="yuv420", should_stop=lambda: STOP_FLAG):
            recon_writer.write_one_frame(frame.y, frame.uv)

            frame_idx += 1
            now = time.monotonic()
            current_fps = rolling_fps.record_frame(now)
            if (now - t_last) >= 0.1:
                progress_q.put({
                    "type": "worker_prog",
                    "wid": wid,
                    "vid": task.base_name,
                    "frames": frame_idx,
                    "frame_total": frame_total,
                    "fps": current_fps,
                })
                t_last = now
    finally:
        recon_writer.close()
        source.close()

    append_progress_log(output_dir, {
        "video_id": task.base_name,
        "status": "decode-done",
        "out_yuv": str(output_path),
        "frames": frame_idx,
    })
    progress_q.put({
        "type": "worker_done",
        "wid": wid,
        "vid": task.base_name,
        "frames": frame_idx,
        "frame_total": frame_total,
        "elapsed": time.monotonic() - t0,
    })


def worker_entry(wid: int, task_q, progress_q, stop_event, args_dict: Dict, use_cuda: bool,
                 cuda_device_index: Optional[int]):
    global STOP_FLAG
    STOP_FLAG = False
    signal.signal(signal.SIGTERM, _worker_sig_handler)
    signal.signal(signal.SIGINT, _worker_sig_handler)

    device = resolve_decode_device(use_cuda, cuda_device_index)
    models = load_decoder_models(
        args_dict["model_path_i"],
        args_dict["model_path_p"],
        device,
        args_dict["force_zero_thres"],
    )
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

        try:
            run_decoding(task, args_dict["output_folder"], progress_q, wid, models)
        except Exception as exc:
            append_progress_log(Path(args_dict["output_folder"]), {
                "video_id": task.base_name,
                "status": "failed",
                "stage": "decode",
                "error": str(exc),
            })
            progress_q.put({
                "type": "worker_fail",
                "wid": wid,
                "vid": task.base_name,
                "error": str(exc),
            })


def configure_parser(parser: argparse.ArgumentParser):
    parser.add_argument("--model_path_i", type=str, default="./checkpoints/cvpr2025_image.pth.tar")
    parser.add_argument("--model_path_p", type=str, default="./checkpoints/cvpr2025_video.pth.tar")
    parser.add_argument("--input_folder", type=str, required=True, help="Folder with .bin files.")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder for decoded .yuv files.")
    parser.add_argument("--original_folder", type=str, default=None, help="[Optional] Folder with original videos for bitrate/frame count.")
    parser.add_argument(
        "--worker",
        "-w",
        type=int,
        default=None,
        help=(
            "Number of independent decode workers. By default, CUDA uses one worker per "
            "selected/visible GPU and CPU mode uses one worker."
        ),
    )
    parser.add_argument("--cuda", type=str2bool, default=True)
    parser.add_argument(
        "--cuda_idx",
        type=int,
        nargs="+",
        help="Logical CUDA device indices to use; defaults to every visible GPU.",
    )
    parser.add_argument("--force_zero_thres", type=float, default=None)


def _format_worker_text(state: Dict) -> str:
    vid = state.get("vid", "-")
    if len(vid) > 30:
        vid = vid[:27] + "..."
    total = state.get("frame_total", 0) or 0
    frames = state.get("frames", 0)
    device = state.get("device", "")
    device_text = f"{device} " if device else ""
    if total > 0:
        return f"{device_text}{vid} {frames}/{total}f {state.get('fps', 0.0):.1f}fps"
    return f"{device_text}{vid} {frames}f {state.get('fps', 0.0):.1f}fps"


def run(args) -> int:
    try:
        device_plan = build_worker_device_plan(
            use_cuda=args.cuda,
            requested_workers=args.worker,
            requested_cuda_indices=args.cuda_idx,
            cuda_available=torch.cuda.is_available(),
            cuda_device_count=torch.cuda.device_count(),
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return 2

    pending_tasks, total_found, already_done = build_decode_tasks(args)
    if total_found == 0:
        print("Nothing to do — no .bin files found.")
        return 0

    if args.cuda and not device_plan.using_cuda:
        print("Warning: --cuda specified but no devices found. Running on CPU.")

    if not pending_tasks:
        progress = MultiWorkerProgress(total_found, 0, "Decode")
        progress.set_done(already_done)
        progress.close()
        print("Nothing to do — all bitstreams are already decoded.")
        return 0

    worker_count = min(device_plan.worker_count, len(pending_tasks))
    cuda_plan = device_plan.cuda_indices[:worker_count]
    if device_plan.using_cuda:
        description = ", ".join(
            f"worker {wid}->cuda:{cuda_plan[wid]}" for wid in range(worker_count)
        )
    else:
        noun = "worker" if worker_count == 1 else "workers"
        description = f"{worker_count} CPU {noun}"
    print(f"Worker plan: {description}")

    ctx = get_context("spawn")
    task_q = ctx.Queue()
    progress_q = ctx.Queue()
    stop_event = ctx.Event()
    args_dict = vars(args).copy()

    workers = []
    for wid in range(worker_count):
        proc = ctx.Process(
            target=worker_entry,
            args=(
                wid,
                task_q,
                progress_q,
                stop_event,
                args_dict,
                device_plan.using_cuda,
                cuda_plan[wid],
            ),
            daemon=True,
        )
        proc.start()
        workers.append(proc)

    for task in pending_tasks:
        task_q.put(task)
    for _ in workers:
        task_q.put(None)

    progress = MultiWorkerProgress(total_found, len(workers), "Decode")
    progress.set_done(already_done)
    active: Dict[int, Dict] = {}

    try:
        alive = True
        while alive or not progress_q.empty():
            try:
                msg = progress_q.get(timeout=0.2)
            except queue.Empty:
                msg = None

            if msg:
                typ = msg.get("type")
                wid = msg.get("wid", 0)
                if typ == "worker_hello":
                    progress.set_worker_text(
                        wid,
                        f"{msg.get('device', 'unknown')} ready (pid {msg.get('pid', '?')})",
                    )
                elif typ == "worker_start":
                    active[wid] = {
                        "vid": msg["vid"],
                        "device": (
                            f"cuda:{cuda_plan[wid]}"
                            if cuda_plan[wid] is not None
                            else "cpu"
                        ),
                        "frames": 0,
                        "frame_total": msg.get("frame_total", 0),
                        "fps": 0.0,
                    }
                    progress.set_worker_text(wid, _format_worker_text(active[wid]))
                elif typ == "worker_prog":
                    state = active.setdefault(wid, {"vid": msg["vid"]})
                    state.update({
                        "frames": msg.get("frames", 0),
                        "frame_total": msg.get("frame_total", 0),
                        "fps": msg.get("fps", 0.0),
                    })
                    progress.set_worker_text(wid, _format_worker_text(state))
                elif typ in {"worker_done", "worker_fail"}:
                    active.pop(wid, None)
                    progress.clear_worker(wid)
                    progress.increment_done(
                        elapsed_seconds=msg.get("elapsed") if typ == "worker_done" else None
                    )

            alive = any(proc.is_alive() for proc in workers)
    finally:
        progress.close()
        stop_event.set()
        for proc in workers:
            proc.join()

    print(f"Decoding finished for {total_found} files.")
    return 0
