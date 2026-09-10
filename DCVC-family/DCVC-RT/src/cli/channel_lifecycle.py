from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from src.cli.progress import encoded_base_from_bin_name, progress_log_path
from src.cli.thumbnail_codec import (
    DEFAULT_THUMBNAIL_EXTENSIONS,
    image_dimensions,
    inspect_intra_image_bitstream,
    thumbnail_output_name,
)
from src.utils.stream_helper import NalType, SPSHelper, read_header, read_sps_remaining, skip_ip_remaining


STATE_FILENAME = ".channel-state.json"
AUDIT_FILENAME = ".cleanup-audit.jsonl"
ARCHIVE_MANIFEST_FILENAME = ".archive-manifest.json"
STATE_SCHEMA_VERSION = 1
ARCHIVE_SCHEMA_VERSION = 1


class LifecycleError(RuntimeError):
    pass


def validate_channel_id(channel_id: str) -> str:
    if (
        not channel_id
        or channel_id in {".", ".."}
        or "/" in channel_id
        or "\\" in channel_id
        or Path(channel_id).name != channel_id
    ):
        raise LifecycleError(f"channel ID must be one directory name: {channel_id!r}")
    return channel_id


@dataclass
class ArtifactRecord:
    kind: str
    path: str
    size: int
    sha256: str
    mtime_ns: int = 0
    device: int = 0
    inode: int = 0


@dataclass
class ItemValidation:
    base: str
    source_path: str
    source_size: int
    already_deleted: bool = False
    artifacts: List[ArtifactRecord] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors


@dataclass
class ThumbnailValidation:
    source_path: str
    source_size: int
    width: int
    height: int
    qp: int
    already_deleted: bool = False
    artifact: Optional[ArtifactRecord] = None
    errors: List[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors


@dataclass
class ChannelValidation:
    channel_id: str
    state_path: str
    eligible: bool
    reclaimable_bytes: int
    retained_bytes: int
    items: List[ItemValidation]
    thumbnails: List[ThumbnailValidation] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    inventory_signature: str = ""

    @property
    def delete_file_count(self) -> int:
        return sum(not item.already_deleted for item in self.items) + sum(
            not item.already_deleted for item in self.thumbnails
        )

    def to_dict(self) -> Dict:
        return {
            "channel_id": self.channel_id,
            "state_path": self.state_path,
            "inventory_signature": self.inventory_signature,
            "eligible": self.eligible,
            "reclaimable_bytes": self.reclaimable_bytes,
            "retained_bytes": self.retained_bytes,
            "errors": list(self.errors),
            "items": [
                {
                    **asdict(item),
                    "valid": item.valid,
                }
                for item in self.items
            ],
            "thumbnails": [
                {**asdict(item), "valid": item.valid}
                for item in self.thumbnails
            ],
        }


@dataclass
class CleanupResult:
    channel_id: str
    success: bool
    dry_run: bool
    deleted_files: int
    reclaimed_bytes: int
    errors: List[str] = field(default_factory=list)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _atomic_write_json(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def _append_audit(output_dir: Path, event: str, **fields):
    output_dir.mkdir(parents=True, exist_ok=True)
    record = {"at": now_iso(), "event": event, **fields}
    data = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
    fd = os.open(output_dir / AUDIT_FILENAME, flags, 0o644)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)


def _load_json(path: Path) -> Dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            value = json.load(f)
    except FileNotFoundError as exc:
        raise LifecycleError(f"state file not found: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise LifecycleError(f"cannot read state file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LifecycleError(f"state file is not a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(metadata: os.stat_result):
    return metadata.st_size, metadata.st_mtime_ns, metadata.st_dev, metadata.st_ino


def _inventory_signature(state: Dict) -> str:
    stable_state = {
        key: state.get(key)
        for key in (
            "schema_version",
            "channel_id",
            "input_dir",
            "output_dir",
            "source_extensions",
            "audio_required",
            "metadata_required",
            "encoder_config",
            "thumbnail_codec",
            "thumbnail_extensions",
            "thumbnail_qp",
            "thumbnails",
            "inventory_errors",
            "items",
        )
    }
    encoded = json.dumps(
        stable_state, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _safe_regular_file(path: Path, root: Path, allowed_suffixes: Optional[set[str]] = None):
    root_real = root.resolve(strict=True)
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise LifecycleError(f"missing file: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise LifecycleError(f"refusing symlink: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise LifecycleError(f"not a regular file: {path}")
    path_real = path.resolve(strict=True)
    if not _is_within(path_real, root_real):
        raise LifecycleError(f"path escapes expected root {root}: {path}")
    if allowed_suffixes is not None and path.suffix.lower() not in allowed_suffixes:
        raise LifecycleError(f"unexpected source extension: {path}")
    return metadata


def _copy_sidecar_atomic(source: Path, destination: Path):
    if source.resolve(strict=True) == destination.resolve(strict=False):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise LifecycleError(f"unsafe retained sidecar destination: {destination}")
        if source.stat().st_size != destination.stat().st_size or _sha256(source) != _sha256(destination):
            raise LifecycleError(f"retained sidecar already exists with different contents: {destination}")
        return
    tmp = destination.with_name(
        f".{destination.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    try:
        with source.open("rb") as src, tmp.open("xb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(tmp, destination)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def state_path_for(output_dir: Path) -> Path:
    return output_dir / STATE_FILENAME


def write_channel_inventory(
    channel_id: str,
    input_dir: Path,
    output_dir: Path,
    source_paths: Sequence[Path],
    source_extensions: Iterable[str],
    audio_required: bool,
    metadata_required: bool,
    encoder_config: Dict,
    thumbnail_sources: Sequence[Path] = (),
    thumbnail_extensions: Iterable[str] = DEFAULT_THUMBNAIL_EXTENSIONS,
    thumbnail_codec: str = "keep",
    thumbnail_qp: int = 45,
) -> Path:
    channel_id = validate_channel_id(channel_id)
    input_dir = input_dir.resolve(strict=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = output_dir.resolve(strict=True)
    path = state_path_for(output_dir)
    if path.exists():
        previous = _load_json(path)
        previous_status = previous.get("cleanup_status")
        if previous_status in {"deleting", "partial-failure"}:
            raise LifecycleError(
                f"channel {channel_id} has an interrupted cleanup ({previous_status}); "
                "resume it with the cleanup command before starting a new inventory"
            )
    allowed = {suffix.lower() for suffix in source_extensions}
    thumbnail_allowed = {
        suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
        for suffix in thumbnail_extensions
    }
    if thumbnail_codec not in {"keep", "dcvc-intra"}:
        raise LifecycleError(f"unsupported thumbnail codec: {thumbnail_codec}")
    items = []
    seen_bases: Dict[str, str] = {}
    inventory_errors = []

    for source in sorted((Path(path) for path in source_paths), key=lambda value: str(value)):
        source = source.resolve(strict=False)
        base = source.stem
        try:
            metadata = _safe_regular_file(source, input_dir, allowed)
        except LifecycleError as exc:
            inventory_errors.append(str(exc))
            continue
        if base in seen_bases:
            inventory_errors.append(
                f"duplicate source basename {base!r}: {seen_bases[base]} and {source}"
            )
        seen_bases[base] = str(source)
        sidecars = []
        if thumbnail_codec == "keep":
            for suffix in DEFAULT_THUMBNAIL_EXTENSIONS:
                candidate = source.with_name(base + suffix)
                if candidate.is_file() and not candidate.is_symlink():
                    sidecars.append(str(candidate.absolute()))
        items.append({
            "base": base,
            "source_path": str(source),
            "source_size": metadata.st_size,
            "source_mtime_ns": metadata.st_mtime_ns,
            "source_device": metadata.st_dev,
            "source_inode": metadata.st_ino,
            "metadata_source": str(source.with_name(base + ".info.json").absolute()),
            "sidecar_sources": sidecars,
        })

    thumbnails = []
    seen_thumbnail_sources = set()
    if thumbnail_codec == "dcvc-intra":
        for source in sorted((Path(value) for value in thumbnail_sources), key=lambda value: str(value)):
            source = source.resolve(strict=False)
            if str(source) in seen_thumbnail_sources:
                inventory_errors.append(f"duplicate thumbnail source path: {source}")
                continue
            seen_thumbnail_sources.add(str(source))
            try:
                metadata = _safe_regular_file(source, input_dir, thumbnail_allowed)
                width, height = image_dimensions(source)
            except (LifecycleError, OSError, ValueError) as exc:
                inventory_errors.append(str(exc))
                continue
            thumbnails.append({
                "source_path": str(source),
                "source_size": metadata.st_size,
                "source_mtime_ns": metadata.st_mtime_ns,
                "source_device": metadata.st_dev,
                "source_inode": metadata.st_ino,
                "width": width,
                "height": height,
                "qp": int(thumbnail_qp),
                "artifact_name": thumbnail_output_name(source, thumbnail_qp),
            })

    payload = {
        "schema_version": STATE_SCHEMA_VERSION,
        "created_at": now_iso(),
        "channel_id": channel_id,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "source_extensions": sorted(allowed),
        "audio_required": bool(audio_required),
        "metadata_required": bool(metadata_required),
        "encoder_config": dict(encoder_config),
        "thumbnail_codec": thumbnail_codec,
        "thumbnail_extensions": sorted(thumbnail_allowed),
        "thumbnail_qp": int(thumbnail_qp) if thumbnail_codec == "dcvc-intra" else None,
        "thumbnails": thumbnails,
        "inventory_errors": inventory_errors,
        "cleanup_status": "encoding",
        "items": items,
    }
    _atomic_write_json(path, payload)
    _append_audit(
        output_dir,
        "inventory-written",
        channel_id=channel_id,
        sources=len(items),
        thumbnails=len(thumbnails),
        errors=len(inventory_errors),
    )
    return path


def _progress_details(output_dir: Path):
    completed_records: Dict[str, Dict] = {}
    current_status: Dict[str, str] = {}
    frame_counts: Dict[str, int] = {}
    log_path = progress_log_path(output_dir)
    if not log_path.exists():
        return {}, frame_counts
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            base = record.get("video_id")
            if not isinstance(base, str) or not base:
                continue
            status = record.get("status")
            if status in {"started", "video-encoding"}:
                current_status[base] = "incomplete"
            elif status == "video-done":
                current_status[base] = "done"
                completed_records[base] = record
            elif status == "failed" and record.get("stage") == "video":
                current_status[base] = "failed"
            if record.get("stage") == "stats" and record.get("event") == "frames_encoded":
                try:
                    frame_counts[base] = int(record["frames"])
                except (KeyError, TypeError, ValueError):
                    pass
    completed = {
        base: record for base, record in completed_records.items()
        if current_status.get(base) == "done"
    }
    return completed, frame_counts


def _deleted_sources(output_dir: Path) -> set[str]:
    deleted = set()
    audit_path = output_dir / AUDIT_FILENAME
    if not audit_path.exists():
        return deleted
    with audit_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") == "source-deleted" and isinstance(record.get("source_path"), str):
                deleted.add(record["source_path"])
    return deleted


def _deletion_intents(output_dir: Path) -> set[str]:
    intents = set()
    audit_path = output_dir / AUDIT_FILENAME
    if not audit_path.exists():
        return intents
    with audit_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") == "source-delete-intent" and isinstance(record.get("source_path"), str):
                intents.add(record["source_path"])
            elif record.get("event") == "source-deleted" and isinstance(record.get("source_path"), str):
                intents.discard(record["source_path"])
    return intents


def _resolve_output_artifact(value: str, output_dir: Path) -> Path:
    # Progress logs can be shared by machines that mount the same channel at
    # different absolute paths. Final encode artifacts always live directly in
    # the current channel output directory, so only the recorded filename is
    # portable and authoritative here.
    path = Path(value)
    if not path.name or path.name in {".", ".."}:
        raise LifecycleError(f"invalid output artifact path: {value!r}")
    return (output_dir / path.name).absolute()


def _probe_opus(path: Path) -> float:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,duration:format=duration",
        "-of", "json", str(path),
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise LifecycleError(f"ffprobe timed out for {path}") from exc
    if result.returncode != 0:
        raise LifecycleError(f"ffprobe rejected {path}: {result.stderr.strip() or 'unknown error'}")
    try:
        payload = json.loads(result.stdout)
        streams = payload.get("streams") or []
        if not streams or streams[0].get("codec_name") != "opus":
            raise ValueError("no Opus audio stream")
        duration_raw = streams[0].get("duration") or payload.get("format", {}).get("duration")
        duration = float(duration_raw)
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise LifecycleError(f"invalid Opus probe for {path}: {exc}") from exc
    if duration <= 0:
        raise LifecycleError(f"non-positive Opus duration for {path}")
    return duration


def _artifact(kind: str, path: Path, output_root: Path) -> ArtifactRecord:
    before = _safe_regular_file(path, output_root)
    if before.st_size <= 0:
        raise LifecycleError(f"empty retained artifact: {path}")
    sha256 = _sha256(path)
    after = _safe_regular_file(path, output_root)
    if _file_identity(before) != _file_identity(after):
        raise LifecycleError(f"retained artifact changed during validation: {path}")
    return ArtifactRecord(
        kind=kind,
        path=str(path),
        size=after.st_size,
        sha256=sha256,
        mtime_ns=after.st_mtime_ns,
        device=after.st_dev,
        inode=after.st_ino,
    )


def _inspect_bitstream_strict(path: Path):
    file_size = path.stat().st_size
    frame_count = 0
    width = 0
    height = 0
    sps_helper = SPSHelper()
    with path.open("rb") as f:
        while f.tell() < file_size:
            offset = f.tell()
            try:
                header = read_header(f)
                nal_type = header["nal_type"]
                if nal_type == NalType.NAL_SPS:
                    sps = read_sps_remaining(f, header["sps_id"])
                    if sps["width"] <= 0 or sps["height"] <= 0:
                        raise LifecycleError(f"invalid SPS dimensions at byte {offset}")
                    sps_helper.add_sps_by_id(sps)
                    if width <= 0 or height <= 0:
                        width, height = sps["width"], sps["height"]
                elif nal_type in (NalType.NAL_I, NalType.NAL_P):
                    if sps_helper.get_sps_by_id(header["sps_id"]) is None:
                        raise LifecycleError(f"frame at byte {offset} references a missing SPS")
                    skip_ip_remaining(f)
                    if f.tell() > file_size:
                        raise LifecycleError(f"truncated frame payload at byte {offset}")
                    frame_count += 1
                else:
                    raise LifecycleError(f"unsupported NAL type at byte {offset}: {nal_type}")
            except LifecycleError:
                raise
            except Exception as exc:
                raise LifecycleError(f"malformed DCVC bitstream at byte {offset}: {exc}") from exc
        if f.tell() != file_size:
            raise LifecycleError(f"bitstream parser did not end at EOF: {path}")
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise LifecycleError(f"invalid or empty DCVC bitstream: {path}")
    return frame_count, width, height


def validate_channel(state_path: Path) -> ChannelValidation:
    state_path = Path(state_path).absolute()
    state = _load_json(state_path)
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise LifecycleError(f"unsupported state schema in {state_path}")

    channel_id = validate_channel_id(str(state.get("channel_id", "")))
    input_dir = Path(state["input_dir"]).resolve(strict=True)
    output_dir = Path(state["output_dir"]).resolve(strict=True)
    if state_path.resolve(strict=True) != state_path_for(output_dir):
        raise LifecycleError(f"state file is outside its declared output directory: {state_path}")
    if input_dir.name != channel_id or output_dir.name != channel_id:
        raise LifecycleError(
            f"channel directories do not match channel ID {channel_id!r}: {input_dir}, {output_dir}"
        )
    allowed = {str(value).lower() for value in state.get("source_extensions", [])}
    thumbnail_allowed = {
        str(value).lower() for value in state.get("thumbnail_extensions", [])
    }
    metadata_required = bool(state.get("metadata_required", True))
    audio_required = bool(state.get("audio_required", True))
    completed, frame_counts = _progress_details(output_dir)
    already_deleted = _deleted_sources(output_dir)
    pending_delete_intents = _deletion_intents(output_dir)
    validation_errors = list(state.get("inventory_errors") or [])
    item_results: List[ItemValidation] = []
    thumbnail_results: List[ThumbnailValidation] = []

    for item in state.get("items") or []:
        base = str(item.get("base", ""))
        source = Path(item.get("source_path", ""))
        result = ItemValidation(
            base=base,
            source_path=str(source),
            source_size=int(item.get("source_size", 0)),
            already_deleted=(
                str(source) in already_deleted
                or (str(source) in pending_delete_intents and not source.exists())
            ),
        )

        if result.already_deleted:
            if source.exists():
                result.errors.append(f"audited deleted source exists again: {source}")
        else:
            try:
                metadata = _safe_regular_file(source, input_dir, allowed)
                expected = (
                    int(item["source_size"]), int(item["source_mtime_ns"]),
                    int(item["source_device"]), int(item["source_inode"]),
                )
                actual = (metadata.st_size, metadata.st_mtime_ns, metadata.st_dev, metadata.st_ino)
                if actual != expected:
                    result.errors.append(f"source changed since discovery: {source}")
            except (LifecycleError, KeyError, TypeError, ValueError) as exc:
                result.errors.append(str(exc))

        metadata_source = Path(item.get("metadata_source", ""))
        metadata_destination = output_dir / f"{base}.info.json"
        if metadata_source.is_file() and not metadata_source.is_symlink():
            try:
                _copy_sidecar_atomic(metadata_source, metadata_destination)
                with metadata_destination.open("r", encoding="utf-8") as f:
                    metadata_json = json.load(f)
                if not isinstance(metadata_json, dict):
                    raise LifecycleError(f"metadata is not a JSON object: {metadata_destination}")
                result.artifacts.append(_artifact("metadata", metadata_destination, output_dir))
            except (OSError, json.JSONDecodeError, LifecycleError) as exc:
                result.errors.append(str(exc))
        elif metadata_required:
            result.errors.append(f"missing required metadata: {metadata_source}")

        for sidecar_value in item.get("sidecar_sources") or []:
            sidecar_source = Path(sidecar_value)
            sidecar_destination = output_dir / sidecar_source.name
            try:
                _safe_regular_file(sidecar_source, input_dir)
                _copy_sidecar_atomic(sidecar_source, sidecar_destination)
                result.artifacts.append(_artifact("thumbnail", sidecar_destination, output_dir))
            except LifecycleError as exc:
                result.errors.append(str(exc))

        done_record = completed.get(base)
        if not done_record or not isinstance(done_record.get("out_bin"), str):
            result.errors.append(f"no completed bitstream record for {base}")
        else:
            try:
                bin_path = _resolve_output_artifact(done_record["out_bin"], output_dir)
                artifact_base = encoded_base_from_bin_name(bin_path.name)
                if artifact_base != base:
                    raise LifecycleError(f"bitstream filename does not match source {base}: {bin_path}")
                bitstream_artifact = _artifact("bitstream", bin_path, output_dir)
                frame_count, _, _ = _inspect_bitstream_strict(bin_path)
                expected_frames = frame_counts.get(base)
                if expected_frames is not None and expected_frames != frame_count:
                    raise LifecycleError(
                        f"bitstream frame count mismatch for {base}: expected {expected_frames}, got {frame_count}"
                    )
                result.artifacts.append(bitstream_artifact)
            except (OSError, LifecycleError, ValueError) as exc:
                result.errors.append(str(exc))

        if audio_required:
            opus_path = output_dir / f"{base}.opus"
            try:
                opus_artifact = _artifact("audio", opus_path, output_dir)
                _probe_opus(opus_path)
                result.artifacts.append(opus_artifact)
            except (OSError, LifecycleError) as exc:
                result.errors.append(str(exc))

        item_results.append(result)

    for item in state.get("thumbnails") or []:
        source = Path(item.get("source_path", ""))
        result = ThumbnailValidation(
            source_path=str(source),
            source_size=int(item.get("source_size", 0)),
            width=int(item.get("width", 0)),
            height=int(item.get("height", 0)),
            qp=int(item.get("qp", -1)),
            already_deleted=(
                str(source) in already_deleted
                or (str(source) in pending_delete_intents and not source.exists())
            ),
        )
        if result.already_deleted:
            if source.exists():
                result.errors.append(f"audited deleted thumbnail exists again: {source}")
        else:
            try:
                metadata = _safe_regular_file(source, input_dir, thumbnail_allowed)
                expected = (
                    int(item["source_size"]), int(item["source_mtime_ns"]),
                    int(item["source_device"]), int(item["source_inode"]),
                )
                if _file_identity(metadata) != expected:
                    raise LifecycleError(f"thumbnail changed since discovery: {source}")
                if image_dimensions(source) != (result.width, result.height):
                    raise LifecycleError(f"thumbnail dimensions changed since discovery: {source}")
            except (LifecycleError, KeyError, OSError, TypeError, ValueError) as exc:
                result.errors.append(str(exc))

        try:
            artifact_name = Path(str(item["artifact_name"])).name
            if artifact_name != str(item["artifact_name"]):
                raise LifecycleError(f"invalid thumbnail artifact name: {item['artifact_name']!r}")
            artifact_path = output_dir / artifact_name
            info = inspect_intra_image_bitstream(artifact_path)
            if (info.width, info.height) != (result.width, result.height):
                raise LifecycleError(
                    f"thumbnail bitstream dimensions mismatch for {source.name}: "
                    f"expected {result.width}x{result.height}, got {info.width}x{info.height}"
                )
            if info.qp != result.qp:
                raise LifecycleError(
                    f"thumbnail bitstream QP mismatch for {source.name}: "
                    f"expected {result.qp}, got {info.qp}"
                )
            result.artifact = _artifact("thumbnail-intra", artifact_path, output_dir)
        except (KeyError, OSError, LifecycleError, TypeError, ValueError) as exc:
            result.errors.append(str(exc))
        thumbnail_results.append(result)

    if not item_results and not thumbnail_results:
        validation_errors.append("channel inventory contains no source videos or thumbnails")
    for item in item_results:
        validation_errors.extend(f"{item.base}: {error}" for error in item.errors)
    for item in thumbnail_results:
        validation_errors.extend(
            f"{Path(item.source_path).name}: {error}" for error in item.errors
        )
    reclaimable = sum(
        item.source_size for item in item_results if item.valid and not item.already_deleted
    ) + sum(
        item.source_size for item in thumbnail_results if item.valid and not item.already_deleted
    )
    retained = sum(
        artifact.size for item in item_results for artifact in item.artifacts
    ) + sum(
        item.artifact.size for item in thumbnail_results if item.artifact is not None
    )
    validation = ChannelValidation(
        channel_id=channel_id,
        state_path=str(state_path),
        inventory_signature=_inventory_signature(state),
        eligible=not validation_errors,
        reclaimable_bytes=reclaimable,
        retained_bytes=retained,
        items=item_results,
        thumbnails=thumbnail_results,
        errors=validation_errors,
    )
    _append_audit(
        output_dir,
        "validation-passed" if validation.eligible else "validation-failed",
        channel_id=channel_id,
        sources=len(item_results),
        thumbnails=len(thumbnail_results),
        reclaimable_bytes=reclaimable,
        retained_bytes=retained,
        errors=validation.errors,
    )
    return validation


def _snapshot_validation_errors(
    state_path: Path,
    state: Dict,
    validation: ChannelValidation,
) -> List[str]:
    errors: List[str] = []
    if not validation.eligible:
        return list(validation.errors) or ["cached channel validation was not eligible"]
    if Path(validation.state_path).absolute() != state_path:
        errors.append("cached validation belongs to a different channel state file")
    if validation.inventory_signature != _inventory_signature(state):
        errors.append("channel inventory changed after validation")

    channel_id = validate_channel_id(str(state.get("channel_id", "")))
    if validation.channel_id != channel_id:
        errors.append("cached validation belongs to a different channel")
    input_dir = Path(state["input_dir"]).resolve(strict=True)
    output_dir = Path(state["output_dir"]).resolve(strict=True)
    allowed = {str(value).lower() for value in state.get("source_extensions", [])}
    thumbnail_allowed = {
        str(value).lower() for value in state.get("thumbnail_extensions", [])
    }
    state_items = {
        str(Path(item.get("source_path", ""))): item
        for item in state.get("items") or []
    }
    if len(state_items) != len(validation.items):
        errors.append("channel source inventory count changed after validation")
    state_thumbnails = {
        str(Path(item.get("source_path", ""))): item
        for item in state.get("thumbnails") or []
    }
    if len(state_thumbnails) != len(validation.thumbnails):
        errors.append("channel thumbnail inventory count changed after validation")

    completed, _ = _progress_details(output_dir)
    for validated_item in validation.items:
        source = Path(validated_item.source_path)
        item_state = state_items.get(str(source))
        if item_state is None:
            errors.append(f"source left the channel inventory after validation: {source}")
        elif validated_item.already_deleted:
            if source.exists():
                errors.append(f"audited deleted source exists again: {source}")
        else:
            try:
                metadata = _safe_regular_file(source, input_dir, allowed)
                expected = (
                    int(item_state["source_size"]),
                    int(item_state["source_mtime_ns"]),
                    int(item_state["source_device"]),
                    int(item_state["source_inode"]),
                )
                if _file_identity(metadata) != expected:
                    raise LifecycleError(f"source changed after validation: {source}")
            except (LifecycleError, KeyError, TypeError, ValueError) as exc:
                errors.append(str(exc))

        bitstreams = [
            artifact for artifact in validated_item.artifacts if artifact.kind == "bitstream"
        ]
        done_record = completed.get(validated_item.base)
        if len(bitstreams) != 1 or not done_record or not isinstance(done_record.get("out_bin"), str):
            errors.append(f"completed bitstream record changed after validation: {validated_item.base}")
        elif Path(done_record["out_bin"]).name != Path(bitstreams[0].path).name:
            errors.append(f"completed bitstream path changed after validation: {validated_item.base}")

        for artifact in validated_item.artifacts:
            path = Path(artifact.path)
            try:
                metadata = _safe_regular_file(path, output_dir)
                expected = (artifact.size, artifact.mtime_ns, artifact.device, artifact.inode)
                if _file_identity(metadata) != expected:
                    raise LifecycleError(f"retained artifact changed after validation: {path}")
            except (AttributeError, LifecycleError, OSError) as exc:
                errors.append(str(exc))

    for validated_item in validation.thumbnails:
        source = Path(validated_item.source_path)
        item_state = state_thumbnails.get(str(source))
        if item_state is None:
            errors.append(f"thumbnail left the channel inventory after validation: {source}")
        elif validated_item.already_deleted:
            if source.exists():
                errors.append(f"audited deleted thumbnail exists again: {source}")
        else:
            try:
                metadata = _safe_regular_file(source, input_dir, thumbnail_allowed)
                expected = (
                    int(item_state["source_size"]),
                    int(item_state["source_mtime_ns"]),
                    int(item_state["source_device"]),
                    int(item_state["source_inode"]),
                )
                if _file_identity(metadata) != expected:
                    raise LifecycleError(f"thumbnail changed after validation: {source}")
            except (LifecycleError, KeyError, TypeError, ValueError) as exc:
                errors.append(str(exc))

        artifact = validated_item.artifact
        if artifact is None:
            errors.append(f"validated thumbnail has no retained artifact: {source}")
            continue
        if item_state is not None and Path(artifact.path).name != item_state.get("artifact_name"):
            errors.append(f"thumbnail artifact path changed after validation: {source}")
        try:
            metadata = _safe_regular_file(Path(artifact.path), output_dir)
            expected = (artifact.size, artifact.mtime_ns, artifact.device, artifact.inode)
            if _file_identity(metadata) != expected:
                raise LifecycleError(f"retained artifact changed after validation: {artifact.path}")
        except (LifecycleError, OSError) as exc:
            errors.append(str(exc))

    claims_dir = output_dir / ".dcvc-shared-work" / "claims"
    try:
        active_claims = [entry for entry in claims_dir.iterdir() if (entry / "state.json").exists()]
    except FileNotFoundError:
        active_claims = []
    except OSError as exc:
        errors.append(f"cannot check shared-work claims before cleanup: {exc}")
        active_claims = []
    if active_claims:
        errors.append(
            f"{len(active_claims)} shared-work claim(s) are still active; wait for all encoders to finish"
        )
    return errors


def _write_archive_manifest(state: Dict, validation: ChannelValidation, output_dir: Path) -> Path:
    payload = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "created_at": now_iso(),
        "channel_id": validation.channel_id,
        "encoder_config": state.get("encoder_config") or {},
        "source_count": len(validation.items),
        "thumbnail_source_count": len(validation.thumbnails),
        "original_bytes": (
            sum(item.source_size for item in validation.items)
            + sum(item.source_size for item in validation.thumbnails)
        ),
        "retained_bytes": validation.retained_bytes,
        "items": [asdict(item) for item in validation.items],
        "thumbnails": [asdict(item) for item in validation.thumbnails],
    }
    path = output_dir / ARCHIVE_MANIFEST_FILENAME
    _atomic_write_json(path, payload)
    return path


def mark_channel_kept(state_path: Path, actor: str = "user"):
    state = _load_json(Path(state_path))
    output_dir = Path(state["output_dir"])
    state["cleanup_status"] = "kept"
    state["updated_at"] = now_iso()
    _atomic_write_json(Path(state_path), state)
    _append_audit(output_dir, "originals-kept", channel_id=state.get("channel_id"), actor=actor)


def record_lifecycle_status(state_path: Path, status: str, event: str, **fields):
    state_path = Path(state_path)
    state = _load_json(state_path)
    output_dir = Path(state["output_dir"])
    state["cleanup_status"] = status
    state["updated_at"] = now_iso()
    _atomic_write_json(state_path, state)
    _append_audit(
        output_dir,
        event,
        channel_id=state.get("channel_id"),
        status=status,
        **fields,
    )


def cleanup_channel(
    state_path: Path,
    dry_run: bool = False,
    actor: str = "user",
    prevalidated: Optional[ChannelValidation] = None,
) -> CleanupResult:
    state_path = Path(state_path).absolute()
    state = _load_json(state_path)
    output_dir = Path(state["output_dir"])
    channel_id = str(state.get("channel_id", ""))
    validation = prevalidated if prevalidated is not None else validate_channel(state_path)
    validation_errors = _snapshot_validation_errors(state_path, state, validation)
    if prevalidated is not None:
        _append_audit(
            output_dir,
            "validation-snapshot-confirmed" if not validation_errors else "validation-snapshot-invalidated",
            channel_id=channel_id,
            sources=len(validation.items),
            thumbnails=len(validation.thumbnails),
            errors=validation_errors,
        )
    if validation_errors:
        state["cleanup_status"] = "cleanup-failed"
        state["updated_at"] = now_iso()
        _atomic_write_json(state_path, state)
        return CleanupResult(
            channel_id=channel_id,
            success=False,
            dry_run=dry_run,
            deleted_files=0,
            reclaimed_bytes=0,
            errors=validation_errors,
        )

    manifest_path = _write_archive_manifest(state, validation, output_dir)
    _append_audit(
        output_dir,
        "cleanup-authorized",
        channel_id=channel_id,
        actor=actor,
        dry_run=dry_run,
        manifest=str(manifest_path),
        files=validation.delete_file_count,
        reclaimable_bytes=validation.reclaimable_bytes,
    )
    if dry_run:
        state["cleanup_status"] = "dry-run-complete"
        state["updated_at"] = now_iso()
        _atomic_write_json(state_path, state)
        return CleanupResult(
            channel_id=channel_id,
            success=True,
            dry_run=True,
            deleted_files=0,
            reclaimed_bytes=0,
        )

    state["cleanup_status"] = "deleting"
    state["updated_at"] = now_iso()
    _atomic_write_json(state_path, state)

    deleted_files = 0
    reclaimed_bytes = 0
    deleted_before = _deleted_sources(output_dir)
    validated_sources = [*validation.items, *validation.thumbnails]
    for validated_item in validated_sources:
        if validated_item.already_deleted and validated_item.source_path not in deleted_before:
            _append_audit(
                output_dir,
                "source-deleted",
                channel_id=channel_id,
                source_path=validated_item.source_path,
                source_size=validated_item.source_size,
                actor="interrupted-cleanup-recovery",
            )
            deleted_before.add(validated_item.source_path)
    item_by_path = {item.source_path: item for item in validated_sources}
    video_allowed = {str(value).lower() for value in state.get("source_extensions", [])}
    thumbnail_allowed = {
        str(value).lower() for value in state.get("thumbnail_extensions", [])
    }
    input_dir = Path(state["input_dir"])
    cleanup_inventory = [
        (item, video_allowed) for item in state.get("items") or []
    ] + [
        (item, thumbnail_allowed) for item in state.get("thumbnails") or []
    ]
    for item_state, allowed in cleanup_inventory:
        source = Path(item_state["source_path"])
        source_string = str(source)
        if source_string in deleted_before:
            continue
        validated_item = item_by_path[source_string]
        try:
            metadata = _safe_regular_file(source, input_dir, allowed)
            expected = (
                int(item_state["source_size"]), int(item_state["source_mtime_ns"]),
                int(item_state["source_device"]), int(item_state["source_inode"]),
            )
            actual = (metadata.st_size, metadata.st_mtime_ns, metadata.st_dev, metadata.st_ino)
            if actual != expected:
                raise LifecycleError(f"source changed immediately before deletion: {source}")
            parent_fd = os.open(source.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                current = os.stat(source.name, dir_fd=parent_fd, follow_symlinks=False)
                actual = (current.st_size, current.st_mtime_ns, current.st_dev, current.st_ino)
                if not stat.S_ISREG(current.st_mode) or actual != expected:
                    raise LifecycleError(f"source changed during deletion authorization: {source}")
                _append_audit(
                    output_dir,
                    "source-delete-intent",
                    channel_id=channel_id,
                    source_path=source_string,
                    source_size=validated_item.source_size,
                    actor=actor,
                )
                os.unlink(source.name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
            deleted_files += 1
            reclaimed_bytes += validated_item.source_size
            _append_audit(
                output_dir,
                "source-deleted",
                channel_id=channel_id,
                source_path=source_string,
                source_size=validated_item.source_size,
                actor=actor,
            )
        except (OSError, LifecycleError, KeyError, TypeError, ValueError) as exc:
            errors = [str(exc)]
            _append_audit(
                output_dir,
                "cleanup-failed",
                channel_id=channel_id,
                source_path=source_string,
                errors=errors,
            )
            state["cleanup_status"] = "partial-failure"
            state["updated_at"] = now_iso()
            _atomic_write_json(state_path, state)
            return CleanupResult(
                channel_id=channel_id,
                success=False,
                dry_run=False,
                deleted_files=deleted_files,
                reclaimed_bytes=reclaimed_bytes,
                errors=errors,
            )

    state["cleanup_status"] = "deleted"
    state["updated_at"] = now_iso()
    state["deleted_files"] = len(validated_sources)
    state["reclaimed_bytes"] = sum(item.source_size for item in validated_sources)
    _atomic_write_json(state_path, state)
    _append_audit(
        output_dir,
        "cleanup-complete",
        channel_id=channel_id,
        deleted_files=deleted_files,
        reclaimed_bytes=reclaimed_bytes,
        actor=actor,
    )
    return CleanupResult(
        channel_id=channel_id,
        success=True,
        dry_run=False,
        deleted_files=deleted_files,
        reclaimed_bytes=reclaimed_bytes,
    )


def find_channel_state(output_root: Path, channel_id: str) -> Path:
    return state_path_for(Path(output_root) / validate_channel_id(channel_id))
