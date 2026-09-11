from __future__ import annotations

import io
import json
import os
import queue
import signal
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from src.cli.progress import append_progress_log
from src.layers.cuda_inference import replicate_pad
from src.layers.int16_inference import residual_autotune_scope
from src.models.image_model import DMCI
from src.utils.common import load_model_for_inference, set_torch_env
from src.utils.stream_helper import (
    NalType,
    read_header,
    read_ip_remaining,
    read_sps_remaining,
    write_ip,
    write_sps,
)
from src.utils.transforms import rgb2ycbcr


THUMBNAIL_PROGRESS_FILENAME = ".thumbnail-progress.jsonl"
DEFAULT_THUMBNAIL_EXTENSIONS = (".webp", ".jpg", ".jpeg", ".png")
STOP_FLAG = False


@dataclass(frozen=True)
class IntraImageInfo:
    width: int
    height: int
    qp: int
    ec_part: int
    payload_size: int


@dataclass(frozen=True)
class ThumbnailTask:
    channel_id: str
    source_path: str
    source_relative: str
    output_path: str
    source_size: int
    source_mtime_ns: int
    source_device: int
    source_inode: int
    width: int
    height: int
    qp: int


@dataclass
class ThumbnailDiscovery:
    channel_id: str
    source_paths: List[str]
    pending_tasks: List[ThumbnailTask]
    total_found: int
    already_done: int
    errors: List[str]


def thumbnail_output_name(source: Path | str, qp: int) -> str:
    return f"{Path(source).name}_qI{int(qp)}.dcvci"


def image_dimensions(path: Path | str) -> Tuple[int, int]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image dimensions: {path}")
    return int(width), int(height)


def inspect_intra_image_bitstream(path: Path | str) -> IntraImageInfo:
    """Strictly parse a standalone SPS + one I-frame image bitstream."""

    path = Path(path)
    file_size = path.stat().st_size
    if file_size <= 0:
        raise ValueError(f"empty DCVC intra image: {path}")

    sps = None
    frame = None
    with path.open("rb") as stream:
        while stream.tell() < file_size:
            offset = stream.tell()
            try:
                header = read_header(stream)
                if header["nal_type"] == NalType.NAL_SPS:
                    if sps is not None or frame is not None:
                        raise ValueError("expected exactly one SPS before the image frame")
                    sps = read_sps_remaining(stream, header["sps_id"])
                    if int(sps["width"]) <= 0 or int(sps["height"]) <= 0:
                        raise ValueError("invalid SPS dimensions")
                    continue
                if header["nal_type"] != NalType.NAL_I:
                    raise ValueError(f"expected an I-frame, found {header['nal_type']}")
                if sps is None or header["sps_id"] != sps["sps_id"]:
                    raise ValueError("image frame references a missing SPS")
                if frame is not None:
                    raise ValueError("expected exactly one image frame")
                qp, payload = read_ip_remaining(stream)
                frame = (int(qp), payload)
            except (EOFError, IndexError, KeyError, struct.error) as exc:
                raise ValueError(f"malformed DCVC intra image at byte {offset}: {exc}") from exc
        if stream.tell() != file_size:
            raise ValueError("DCVC intra image parser did not end at EOF")

    if sps is None or frame is None:
        raise ValueError(f"missing SPS or I-frame in DCVC intra image: {path}")
    return IntraImageInfo(
        width=int(sps["width"]),
        height=int(sps["height"]),
        qp=frame[0],
        ec_part=int(sps["ec_part"]),
        payload_size=len(frame[1]),
    )


def _read_completed_records(output_dir: Path) -> Dict[str, Dict]:
    latest: Dict[str, Dict] = {}
    log_path = output_dir / THUMBNAIL_PROGRESS_FILENAME
    if not log_path.exists():
        return latest
    with log_path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            relative = record.get("source_relative")
            if isinstance(relative, str) and relative:
                latest[relative] = record
    return latest


def _source_identity(path: Path) -> Tuple[int, int, int, int]:
    metadata = path.stat()
    return metadata.st_size, metadata.st_mtime_ns, metadata.st_dev, metadata.st_ino


def _record_matches(record: Optional[Dict], task: ThumbnailTask) -> bool:
    if not record or record.get("status") != "done":
        return False
    try:
        identity = (
            int(record["source_size"]),
            int(record["source_mtime_ns"]),
            int(record["source_device"]),
            int(record["source_inode"]),
        )
        return (
            identity == (
                task.source_size,
                task.source_mtime_ns,
                task.source_device,
                task.source_inode,
            )
            and int(record["width"]) == task.width
            and int(record["height"]) == task.height
            and int(record["qp"]) == task.qp
            and Path(str(record["output"])).name == Path(task.output_path).name
        )
    except (KeyError, TypeError, ValueError):
        return False


def _artifact_matches(task: ThumbnailTask) -> bool:
    try:
        info = inspect_intra_image_bitstream(task.output_path)
    except (OSError, ValueError):
        return False
    return (info.width, info.height, info.qp) == (task.width, task.height, task.qp)


def _done_record(task: ThumbnailTask, recovered: bool = False) -> Dict:
    return {
        "source_relative": task.source_relative,
        "status": "done",
        "source_path": task.source_path,
        "source_size": task.source_size,
        "source_mtime_ns": task.source_mtime_ns,
        "source_device": task.source_device,
        "source_inode": task.source_inode,
        "width": task.width,
        "height": task.height,
        "qp": task.qp,
        "output": task.output_path,
        "recovered": recovered,
    }


def discover_thumbnails(
    channel_id: str,
    input_dir: Path,
    output_dir: Path,
    qp: int,
    recursive: bool = False,
    extensions: Iterable[str] = DEFAULT_THUMBNAIL_EXTENSIONS,
) -> ThumbnailDiscovery:
    allowed = {value.lower() for value in extensions}
    if not input_dir.is_dir():
        return ThumbnailDiscovery(channel_id, [], [], 0, 0, [f"input channel not found: {input_dir}"])
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    sources = sorted(
        (path for path in iterator if path.is_file() and path.suffix.lower() in allowed),
        key=lambda path: path.relative_to(input_dir).as_posix(),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _read_completed_records(output_dir)
    tasks: List[ThumbnailTask] = []
    errors: List[str] = []
    output_owners: Dict[str, str] = {}
    already_done = 0

    for source in sources:
        relative = source.relative_to(input_dir).as_posix()
        output_name = thumbnail_output_name(source, qp)
        previous = output_owners.get(output_name)
        if previous is not None:
            errors.append(
                f"thumbnail output collision for {previous!r} and {relative!r}: {output_name}"
            )
            continue
        output_owners[output_name] = relative
        try:
            width, height = image_dimensions(source)
            size, mtime_ns, device, inode = _source_identity(source)
        except (OSError, ValueError) as exc:
            errors.append(f"cannot inspect thumbnail {source}: {exc}")
            continue
        task = ThumbnailTask(
            channel_id=channel_id,
            source_path=str(source.absolute()),
            source_relative=relative,
            output_path=str((output_dir / output_name).absolute()),
            source_size=size,
            source_mtime_ns=mtime_ns,
            source_device=device,
            source_inode=inode,
            width=width,
            height=height,
            qp=int(qp),
        )
        record = records.get(relative)
        if _record_matches(record, task) and _artifact_matches(task):
            already_done += 1
            continue
        # If publication succeeded immediately before a crash, the exact stream
        # can be recovered when the last attempt describes this same source.
        if (
            record
            and record.get("status") == "started"
            and all(record.get(key) == value for key, value in {
                "source_size": size,
                "source_mtime_ns": mtime_ns,
                "source_device": device,
                "source_inode": inode,
                "width": width,
                "height": height,
                "qp": int(qp),
            }.items())
            and _artifact_matches(task)
        ):
            append_progress_log(output_dir, _done_record(task, recovered=True), THUMBNAIL_PROGRESS_FILENAME)
            already_done += 1
            continue
        tasks.append(task)

    return ThumbnailDiscovery(
        channel_id=channel_id,
        source_paths=[str(path.absolute()) for path in sources],
        pending_tasks=tasks,
        total_found=len(sources),
        already_done=already_done,
        errors=errors,
    )


def _atomic_write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _load_image_tensor(task: ThumbnailTask, device: torch.device) -> torch.Tensor:
    with Image.open(task.source_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    if rgb.shape[:2] != (task.height, task.width):
        raise RuntimeError(f"thumbnail dimensions changed while encoding: {task.source_path}")
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device=device, dtype=torch.float32).div_(255.0)
    return rgb2ycbcr(tensor).half()


def _serialize_intra(model: DMCI, task: ThumbnailTask, x: torch.Tensor) -> bytes:
    padding_r, padding_b = DMCI.get_padding_size(task.height, task.width, 16)
    use_two_ec = (task.width * task.height) > (1280 * 720)
    model.set_use_two_entropy_coders(use_two_ec)
    x_padded = replicate_pad(x, padding_b, padding_r).contiguous()
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=False):
        encoded = model.compress(x_padded, task.qp)
    stream = io.BytesIO()
    sps = {
        "sps_id": 0,
        "height": task.height,
        "width": task.width,
        "ec_part": int(use_two_ec),
        "use_ada_i": 0,
    }
    write_sps(stream, sps)
    write_ip(stream, True, 0, task.qp, encoded["bit_stream"])
    return stream.getvalue()


def _round_trip_verify(model: DMCI, path: Path, task: ThumbnailTask):
    info = inspect_intra_image_bitstream(path)
    if (info.width, info.height, info.qp) != (task.width, task.height, task.qp):
        raise RuntimeError(f"published thumbnail header does not match its source: {path}")
    with path.open("rb") as stream:
        header = read_header(stream)
        if header["nal_type"] != NalType.NAL_SPS:
            raise RuntimeError("thumbnail stream does not start with SPS")
        sps = read_sps_remaining(stream, header["sps_id"])
        header = read_header(stream)
        qp, bit_stream = read_ip_remaining(stream)
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=False):
        decoded = model.decompress(bit_stream, sps, qp)["x_hat"]
    if decoded.ndim != 4 or decoded.shape[-2] < task.height or decoded.shape[-1] < task.width:
        raise RuntimeError(f"decoded thumbnail has an invalid tensor shape: {tuple(decoded.shape)}")
    if not bool(torch.isfinite(decoded).all().item()):
        raise RuntimeError("decoded thumbnail contains non-finite values")


def encode_thumbnail(task: ThumbnailTask, model: DMCI):
    source = Path(task.source_path)
    expected = (
        task.source_size,
        task.source_mtime_ns,
        task.source_device,
        task.source_inode,
    )
    if _source_identity(source) != expected:
        raise RuntimeError(f"thumbnail changed since discovery: {source}")
    output_dir = Path(task.output_path).parent
    append_progress_log(
        output_dir,
        {**_done_record(task), "status": "started", "recovered": False},
        THUMBNAIL_PROGRESS_FILENAME,
    )
    x = _load_image_tensor(task, next(model.parameters()).device)
    # Each worker retains its model across images/channels. Only its first image
    # may benchmark new shapes, including shapes used by round-trip verification.
    # A failed encode attempt also consumes this opportunity to avoid repeated
    # tuning spikes on a sequence of failures. Pre-load/source failures do not.
    with residual_autotune_scope(not getattr(model, "_thumbnail_autotune_started", False)):
        model._thumbnail_autotune_started = True
        data = _serialize_intra(model, task, x)
        temporary = Path(task.output_path).with_name(f".{Path(task.output_path).name}.verify.{os.getpid()}")
        try:
            _atomic_write(temporary, data)
            _round_trip_verify(model, temporary, task)
            if _source_identity(source) != expected:
                raise RuntimeError(f"thumbnail changed during encoding: {source}")
            os.replace(temporary, task.output_path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    append_progress_log(output_dir, _done_record(task), THUMBNAIL_PROGRESS_FILENAME)


def _worker_signal_handler(_sig, _frame):
    global STOP_FLAG
    STOP_FLAG = True
    raise SystemExit(130)


def thumbnail_worker_entry(
    wid: int,
    task_queue,
    progress_queue,
    stop_event,
    model_path_i: str,
    force_zero_thres: Optional[float],
    use_cuda: bool,
    cuda_device_index: Optional[int],
):
    global STOP_FLAG
    STOP_FLAG = False
    signal.signal(signal.SIGTERM, _worker_signal_handler)
    signal.signal(signal.SIGINT, _worker_signal_handler)
    set_torch_env()
    if use_cuda and torch.cuda.is_available():
        index = 0 if cuda_device_index is None else int(cuda_device_index)
        torch.cuda.set_device(index)
        device = torch.device(f"cuda:{index}")
    else:
        device = torch.device("cpu")
    model = load_model_for_inference(DMCI(), model_path_i, device, force_zero_thres)
    progress_queue.put({"type": "worker_hello", "wid": wid, "pid": os.getpid(), "device": str(device)})

    while not stop_event.is_set() and not STOP_FLAG:
        try:
            task = task_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if task is None:
            break
        started = time.monotonic()
        progress_queue.put({
            "type": "worker_start",
            "wid": wid,
            "channel_id": task.channel_id,
            "vid": task.source_relative,
            "resolution": f"{task.width}, {task.height} -> {task.width}, {task.height}",
        })
        try:
            encode_thumbnail(task, model)
            progress_queue.put({
                "type": "worker_done", "wid": wid, "channel_id": task.channel_id,
                "vid": task.source_relative, "elapsed": time.monotonic() - started,
            })
        except Exception as exc:  # pylint: disable=broad-exception-caught
            append_progress_log(
                Path(task.output_path).parent,
                {
                    "source_relative": task.source_relative,
                    "status": "failed",
                    "source_path": task.source_path,
                    "error": str(exc),
                },
                THUMBNAIL_PROGRESS_FILENAME,
            )
            progress_queue.put({
                "type": "worker_fail", "wid": wid, "channel_id": task.channel_id,
                "vid": task.source_relative, "error": str(exc),
            })
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
