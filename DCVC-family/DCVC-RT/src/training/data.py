"""One-time media preparation and deterministic, source-disjoint sampling."""

import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from src.utils.transforms import rgb2ycbcr
from src.training.video_data import check_source, index_video, load_video_clip


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".ts"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def natural_key(path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(path))]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _image_size(path):
    with Image.open(path) as image:
        size = image.size
        image.verify()
    if min(size) <= 0:
        raise ValueError(f"invalid image dimensions: {path}")
    return size


def _record(kind, source, frames, frame_format="rgb", dimensions=None):
    source = str(Path(source).resolve())
    if not frames:
        raise ValueError(f"no frames in {source}")
    if dimensions is None:
        dimensions = _image_size(frames[0])
        for frame in frames[1:]:
            if _image_size(frame) != dimensions:
                raise ValueError(f"sequence changes dimensions: {source}")
    stats = [[Path(frame).stat().st_size, Path(frame).stat().st_mtime_ns] for frame in frames]
    return {"version": 1, "id": digest([kind, source]), "source_group": digest(source),
            "source": source, "kind": kind, "format": frame_format,
            "width": dimensions[0], "height": dimensions[1],
            "frames": [str(Path(frame).resolve()) for frame in frames], "frame_stats": stats}


def _scan(path, extensions):
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() not in extensions:
            raise ValueError(f"unsupported source extension: {path}")
        return [path.resolve()]
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sorted((p.resolve() for p in path.rglob("*") if p.is_file() and p.suffix.lower() in extensions),
                  key=natural_key)


def _read_exact(stream, size):
    chunks, remaining = [], size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            if remaining == size:
                return None
            raise ValueError("truncated raw frame from FFmpeg")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _extract_video(path, cache_root, short_edge):
    metadata = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "json", str(path)], check=True, capture_output=True, text=True)
    streams = json.loads(metadata.stdout)["streams"]
    if not streams:
        raise ValueError(f"no video stream in {path}")
    width, height = int(streams[0]["width"]), int(streams[0]["height"])
    if min(width, height) < 1:
        raise ValueError(f"invalid video dimensions: {path}")
    filters = []
    if short_edge is not None:
        ratio = short_edge / min(width, height)
        resized_width, resized_height = max(2, round(width * ratio / 2) * 2), max(2, round(height * ratio / 2) * 2)
        if (resized_width, resized_height) != (width, height):
            filters.append(f"scale={resized_width}:{resized_height}:flags=bicubic")
        width, height = resized_width, resized_height
    elif width % 2 or height % 2:
        width, height = width + width % 2, height + height % 2
        filters.append(f"pad={width}:{height}")
    stat = path.stat()
    identity = {"source": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                "width": width, "height": height, "format": "yuv420_npz", "version": 1}
    cache = Path(cache_root) / digest(identity)
    completion = cache / "complete.json"
    if completion.exists():
        saved = json.loads(completion.read_text())
        frames = [cache / name for name in saved["frames"]]
        if saved["identity"] == identity and all(p.is_file() for p in frames) and \
                saved.get("frame_stats") == [[p.stat().st_size, p.stat().st_mtime_ns] for p in frames]:
            return _record("video", path, frames, "yuv420_npz", (width, height))
        raise ValueError(f"incomplete/corrupt video cache: {cache}; remove it and prepare again")
    Path(cache_root).mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".extract-", dir=cache_root))
    process = None
    try:
        command = ["ffmpeg", "-v", "error", "-nostdin", "-threads", "1", "-noautorotate", "-i", str(path),
                   "-map", "0:v:0", "-an", "-sn", "-dn", "-vsync", "0"]
        if filters:
            command.extend(["-vf", ",".join(filters)])
        command.extend(["-pix_fmt", "yuv420p", "-f", "rawvideo", "pipe:1"])
        names = []
        y_size, uv_size = width * height, width * height // 4
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
            try:
                while True:
                    raw = _read_exact(process.stdout, y_size + 2 * uv_size)
                    if raw is None:
                        break
                    values = np.frombuffer(raw, dtype=np.uint8)
                    name = f"{len(names):08d}.npz"
                    np.savez_compressed(temporary / name,
                                        y=values[:y_size].reshape(height, width),
                                        u=values[y_size:y_size + uv_size].reshape(height // 2, width // 2),
                                        v=values[y_size + uv_size:].reshape(height // 2, width // 2))
                    names.append(name)
                code = process.wait()
                if code:
                    errors.seek(0)
                    raise ValueError(f"FFmpeg failed for {path}: {errors.read(8192).decode(errors='replace')}")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                process.stdout.close()
        if len(names) < 2:
            raise ValueError(f"video has fewer than two frames: {path}")
        if path.stat().st_size != stat.st_size or path.stat().st_mtime_ns != stat.st_mtime_ns:
            raise ValueError(f"source changed during preparation: {path}")
        stats = [[(temporary / name).stat().st_size, (temporary / name).stat().st_mtime_ns] for name in names]
        atomic_json(temporary / "complete.json", {"identity": identity, "frames": names, "frame_stats": stats})
        # Preparation is a single-parent operation. Never overwrite an existing
        # cache produced by another invocation racing this one.
        try:
            temporary.rename(cache)
        except FileExistsError:
            if not completion.exists():
                raise
        return _record("video", path, [cache / name for name in names], "yuv420_npz", (width, height))
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _frame_sequences(root):
    root = Path(root)
    description = root / "description.json"
    if description.is_file():
        raw = json.loads(description.read_text())
        if isinstance(raw, dict) and "seqs" in raw and "frames" in raw:
            for seq in raw["seqs"]:
                directory = root / seq["path"]
                yield directory, [directory / name for name in raw["frames"][:seq["seq_length"]]]
            return
    grouped = {}
    for path in _scan(root, IMAGE_EXTENSIONS):
        grouped.setdefault(path.parent, []).append(path)
    for directory, frames in sorted(grouped.items(), key=lambda item: natural_key(item[0])):
        yield directory, frames


def frame_count(record):
    return record["frame_count"] if record["format"] == "video_direct" else len(record["frames"])


def _index_videos(root, short_edge):
    def prepare(path):
        try:
            record = index_video(path, short_edge)
            record.update(version=1, id=digest(["video", str(path)]), source_group=digest(str(path)),
                          kind="video", format="video_direct", frames=[])
            return path, record
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            return path, exc
    # map preserves source order and therefore split/validation ordering.
    with ThreadPoolExecutor(max_workers=8) as pool:
        yield from pool.map(prepare, _scan(root, VIDEO_EXTENSIONS))


def prepare_data(config, emit=print):
    if not config.data.sources:
        raise ValueError("prepare-training-data requires data.sources")
    records, errors, seen = [], [], set()
    for source in config.data.sources:
        if source.type == "images":
            candidates = ((path, [path]) for path in _scan(source.path, IMAGE_EXTENSIONS))
        elif source.type == "frame_sequences":
            candidates = _frame_sequences(source.path)
        elif config.data.video_loading == "direct":
            candidates = _index_videos(source.path, config.data.resize_short_edge)
        else:
            candidates = ((path, None) for path in _scan(source.path, VIDEO_EXTENSIONS))
        for path, frames in candidates:
            try:
                if source.type == "videos":
                    if config.data.video_loading == "direct":
                        if isinstance(frames, Exception):
                            raise frames
                        record = frames
                    else:
                        record = _extract_video(path, config.data.cache_root, config.data.resize_short_edge)
                else:
                    kind = "image" if source.type == "images" else "video"
                    if kind == "video" and len(frames) < 2:
                        raise ValueError(f"sequence has fewer than two frames: {path}")
                    record = _record(kind, path, frames)
                if record["id"] in seen:
                    raise ValueError(f"duplicate source: {path}")
                seen.add(record["id"])
                record["split"] = source.split
                record["resized"] = source.type == "videos" and config.data.resize_short_edge is not None
                records.append(record)
                action = "Indexed" if record["format"] == "video_direct" else "Prepared"
                emit(f"{action} {path}: {frame_count(record)} frames ({record['width']}x{record['height']})")
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                errors.append({"source": str(path), "error": str(exc)})
                emit(f"Rejected {path}: {exc}")
    # Group assignment is independent of extraction/clip sampling order. Sorting
    # hash scores makes the seeded split stable for identical source inventories.
    assignments = {}
    for record in records:
        group, split = record["source_group"], record["split"]
        if split != "auto":
            if group in assignments and assignments[group] != split:
                raise ValueError(f"source appears in train and validation: {record['source']}")
            assignments[group] = split
    auto_groups = sorted({r["source_group"] for r in records} - set(assignments),
                         key=lambda key: digest([config.seed, key]))
    count = round(len(auto_groups) * config.data.validation_fraction)
    if len(auto_groups) > 1:
        count = min(len(auto_groups) - 1, max(1, count))
    for index, group in enumerate(auto_groups):
        assignments[group] = "validation" if index < count else "train"
    for record in records:
        record["split"] = assignments[record["source_group"]]
    manifest = Path(config.data.manifest)
    report = {"version": 1, "sources": len(records), "rejected": errors, "seed": config.seed,
              "data": config.to_dict()["data"]}
    atomic_json(manifest.with_suffix(".report.json"), report)
    if not records:
        raise ValueError(f"no usable training sources; see {manifest.with_suffix('.report.json')}")
    if {r["split"] for r in records} != {"train", "validation"}:
        raise ValueError("source-disjoint training requires both train and validation sources")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{manifest.name}.", dir=manifest.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, manifest)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return report


def read_manifest(path):
    path = Path(path).resolve()
    records, groups, ids, frame_groups = [], {}, set(), {}
    with open(path, encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            record = json.loads(line)
            required = {"version", "id", "source_group", "split", "kind", "format", "frames", "width", "height"}
            if not required <= record.keys() or record["version"] != 1:
                raise ValueError(f"invalid training manifest record at line {lineno}")
            if record["split"] not in {"train", "validation"} or record["kind"] not in {"image", "video"}:
                raise ValueError(f"invalid split/kind at manifest line {lineno}")
            direct = record["format"] == "video_direct"
            if record["format"] not in {"rgb", "yuv420_npz", "video_direct"} or (not direct and not record["frames"]):
                raise ValueError(f"invalid frame format/list at manifest line {lineno}")
            group, split = record["source_group"], record["split"]
            if group in groups and groups[group] != split:
                raise ValueError(f"source group leaks between splits: {group}")
            groups[group] = split
            if record["id"] in ids:
                raise ValueError(f"duplicate manifest id: {record['id']}")
            ids.add(record["id"])
            if direct:
                required_video = {"source", "source_stat", "frame_count", "fps", "duration", "video_filters"}
                if not required_video <= record.keys() or record["kind"] != "video" or record["frame_count"] < 2:
                    raise ValueError(f"invalid direct video record at manifest line {lineno}")
                source = Path(record["source"]).expanduser()
                source = str((source if source.is_absolute() else path.parent / source).resolve())
                record["source"] = source
                if source in frame_groups and frame_groups[source] != split:
                    raise ValueError(f"video leaks between splits: {source}")
                frame_groups[source] = split
                check_source(record)
            if "frame_stats" in record and len(record["frame_stats"]) != len(record["frames"]):
                raise ValueError(f"frame identity count differs at manifest line {lineno}")
            for frame_index, frame in enumerate(record["frames"]):
                frame = Path(frame).expanduser()
                frame = str((frame if frame.is_absolute() else path.parent / frame).resolve())
                record["frames"][frame_index] = frame
                if frame in frame_groups and frame_groups[frame] != split:
                    raise ValueError(f"frame leaks between splits: {frame}")
                frame_groups[frame] = split
                if not Path(frame).is_file():
                    raise ValueError(f"prepared frame is missing: {frame}")
                if "frame_stats" in record:
                    stat = Path(frame).stat()
                    if record["frame_stats"][frame_index] != [stat.st_size, stat.st_mtime_ns]:
                        raise ValueError(f"prepared frame changed since manifest creation: {frame}")
            records.append(record)
    return records


def load_frame(record, index):
    if record["format"] == "video_direct":
        return load_video_clip(record, index, 1)[0]
    path = record["frames"][index]
    if record["format"] == "rgb":
        with Image.open(path) as image:
            array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        rgb = torch.from_numpy(array).permute(2, 0, 1).float().div_(255)
        return rgb2ycbcr(rgb)
    with np.load(path, allow_pickle=False) as values:
        y = torch.from_numpy(values["y"].copy()).float()
        u = torch.from_numpy(values["u"].copy()).float().repeat_interleave(2, 0).repeat_interleave(2, 1)
        v = torch.from_numpy(values["v"].copy()).float().repeat_interleave(2, 0).repeat_interleave(2, 1)
    return torch.stack((y, u, v)).div_(255)


class TrainingDataset(Dataset):
    """Sampling is a function of seed/epoch/index, independent of worker timing."""

    def __init__(self, records, stage, data, seed, split="train"):
        self.stage, self.data, self.seed, self.split = stage, data, seed, split
        self.epoch = 0
        self.records = [record for record in records if record["split"] == split
                        and (stage.model == "image" or record["kind"] == "video")
                        and frame_count(record) >= stage.sequence_length]
        if not self.records:
            raise ValueError(f"no {split} sources support {stage.name} ({stage.sequence_length} frames)")
        self.length = (stage.samples_per_epoch or len(self.records)) if split == "train" else len(self.records)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rng = random.Random(int(digest([self.seed, self.epoch, index, self.stage.name]), 16))
        record = self.records[index % len(self.records)]
        count = self.stage.sequence_length
        start = rng.randrange(frame_count(record) - count + 1) if self.split == "train" else 0
        if record["format"] == "video_direct":
            frames = load_video_clip(record, start, count)
        else:
            frames = torch.stack([load_frame(record, start + offset) for offset in range(count)])
        if self.data.resize_short_edge is not None and not record.get("resized", False):
            h, w = frames.shape[-2:]
            scale = self.data.resize_short_edge / min(h, w)
            target = (max(1, round(h * scale)), max(1, round(w * scale)))
            if target != (h, w):
                frames = F.interpolate(frames, size=target, mode="bilinear", align_corners=False).clamp(0, 1)
        h, w = self.stage.crop_size
        # Edge padding matches inference; it never stretches smaller sources.
        frames = F.pad(frames, (0, max(0, w - frames.shape[-1]), 0, max(0, h - frames.shape[-2])),
                       mode="replicate")
        max_y, max_x = frames.shape[-2] - h, frames.shape[-1] - w
        top = rng.randint(0, max_y) if self.split == "train" else max_y // 2
        left = rng.randint(0, max_x) if self.split == "train" else max_x // 2
        frames = frames[:, :, top:top + h, left:left + w]
        if self.split == "train" and self.data.horizontal_flip and rng.getrandbits(1):
            frames = frames.flip(-1)
        return frames.contiguous()
