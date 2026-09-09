from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


SHARED_DIRNAME = ".dcvc-shared-work"
SCHEMA_VERSION = 1


class SharedWorkError(RuntimeError):
    pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _canonical_json(value: Dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _atomic_write_json(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as output:
            output.write(_canonical_json(payload) + b"\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while True:
            block = source.read(4 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def pipeline_id(config: Dict) -> str:
    return hashlib.sha256(_canonical_json(config)).hexdigest()


def job_key(video_id: str) -> str:
    return hashlib.sha256(video_id.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ClaimInfo:
    video_id: str
    owner: str
    hostname: str
    pid: int
    age_seconds: float
    progress: Dict

    @property
    def owner_text(self) -> str:
        location = self.hostname or "unknown-host"
        return f"{location} pid {self.pid}" if self.pid else location


class JobLease:
    def __init__(self, pool: "SharedWorkPool", channel_out: Path, video_id: str,
                 claim_path: Path, token: str, directory_fd: int, recovered_stale: bool = False):
        self.pool = pool
        self.channel_out = channel_out
        self.video_id = video_id
        self.claim_path = claim_path
        self.token = token
        self.recovered_stale = recovered_stale
        self._directory_fd = directory_fd
        self._closed = False

    def _owns_path(self) -> bool:
        if self._closed:
            return False
        try:
            opened = os.fstat(self._directory_fd)
            current = self.claim_path.stat()
            if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
                return False
            with (self.claim_path / "state.json").open("r", encoding="utf-8") as source:
                return json.load(source).get("token") == self.token
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return False

    def assert_owned(self):
        if not self._owns_path():
            raise SharedWorkError(
                f"shared claim for {self.video_id!r} was lost; refusing to publish its output"
            )

    def _write_state(self, progress: Dict):
        state = self.pool._claim_payload(self.video_id, self.token, progress or {})
        temporary_name = f".state.tmp.{self.token}"
        fd = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o644,
            dir_fd=self._directory_fd,
        )
        try:
            data = _canonical_json(state) + b"\n"
            offset = 0
            while offset < len(data):
                offset += os.write(fd, data[offset:])
            os.fsync(fd)
        finally:
            os.close(fd)
        os.rename(
            temporary_name,
            "state.json",
            src_dir_fd=self._directory_fd,
            dst_dir_fd=self._directory_fd,
        )

    def heartbeat(self, progress: Optional[Dict] = None):
        self.assert_owned()
        self._write_state(progress or {})
        self.assert_owned()

    def mark_done(self, artifact_names: Iterable[str]):
        self.assert_owned()
        names = sorted(set(artifact_names))
        for name in names:
            if Path(name).name != name:
                raise SharedWorkError(f"invalid shared-work artifact name: {name!r}")
            artifact = self.channel_out / name
            if not artifact.is_file() or artifact.stat().st_size <= 0:
                raise SharedWorkError(f"cannot mark missing or empty artifact complete: {artifact}")
        _atomic_write_json(self.pool.done_path(self.channel_out, self.video_id), {
            "schema_version": SCHEMA_VERSION,
            "pipeline_id": self.pool.pipeline_id_for(self.channel_out),
            "video_id": self.video_id,
            "completed_at": _now_iso(),
            "owner": self.pool.instance_id,
            "artifacts": names,
        })

    def release(self):
        if self._closed:
            return
        try:
            if self._owns_path():
                for name in ("state.json", f".state.tmp.{self.token}"):
                    try:
                        os.unlink(name, dir_fd=self._directory_fd)
                    except FileNotFoundError:
                        pass
                try:
                    self.claim_path.rmdir()
                except FileNotFoundError:
                    pass
        finally:
            os.close(self._directory_fd)
            self._closed = True

    def __enter__(self) -> "JobLease":
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type, exc_value, traceback
        self.release()


class SharedWorkPool:
    def __init__(self, lease_seconds: float = 900.0, instance_name: Optional[str] = None):
        if lease_seconds < 30:
            raise ValueError("shared-work lease must be at least 30 seconds")
        self.lease_seconds = float(lease_seconds)
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        label = instance_name or self.hostname
        self.instance_id = f"{label}:{self.pid}:{uuid.uuid4().hex}"
        self._pipeline_ids: Dict[Path, str] = {}

    @staticmethod
    def root(channel_out: Path) -> Path:
        return Path(channel_out) / SHARED_DIRNAME

    def pipeline_id_for(self, channel_out: Path) -> str:
        channel_out = Path(channel_out).resolve()
        try:
            return self._pipeline_ids[channel_out]
        except KeyError as exc:
            raise SharedWorkError(f"shared pipeline was not initialized for {channel_out}") from exc

    def ensure_pipeline(self, channel_out: Path, config: Dict) -> str:
        channel_out = Path(channel_out).resolve()
        root = self.root(channel_out)
        root.mkdir(parents=True, exist_ok=True)
        (root / "claims").mkdir(exist_ok=True)
        (root / "done").mkdir(exist_ok=True)
        (root / "stale").mkdir(exist_ok=True)
        expected_id = pipeline_id(config)
        path = root / "pipeline.json"
        payload = {
            "schema_version": SCHEMA_VERSION,
            "pipeline_id": expected_id,
            "created_at": _now_iso(),
            "config": config,
        }
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            fd = None
        if fd is not None:
            try:
                data = _canonical_json(payload) + b"\n"
                offset = 0
                while offset < len(data):
                    offset += os.write(fd, data[offset:])
                os.fsync(fd)
            finally:
                os.close(fd)

        existing = None
        for _ in range(50):
            try:
                with path.open("r", encoding="utf-8") as source:
                    existing = json.load(source)
                break
            except (FileNotFoundError, json.JSONDecodeError):
                time.sleep(0.02)
        if existing is None:
            raise SharedWorkError(f"could not read shared pipeline state: {path}")
        if existing.get("pipeline_id") != expected_id:
            raise SharedWorkError(
                f"shared pipeline settings differ for {channel_out}; use the same models and "
                "encode options on every machine, or use a separate output directory"
            )
        self._pipeline_ids[channel_out] = expected_id
        return expected_id

    def claim_path(self, channel_out: Path, video_id: str) -> Path:
        return self.root(channel_out) / "claims" / job_key(video_id)

    def done_path(self, channel_out: Path, video_id: str) -> Path:
        return self.root(channel_out) / "done" / f"{job_key(video_id)}.json"

    def _claim_payload(self, video_id: str, token: str, progress: Dict) -> Dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "video_id": video_id,
            "token": token,
            "owner": self.instance_id,
            "hostname": self.hostname,
            "pid": self.pid,
            "updated_at": _now_iso(),
            "progress": progress,
        }

    def completed_artifacts(self, channel_out: Path, video_id: str) -> Optional[Tuple[str, ...]]:
        path = self.done_path(channel_out, video_id)
        try:
            with path.open("r", encoding="utf-8") as source:
                record = json.load(source)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        if record.get("pipeline_id") != self.pipeline_id_for(channel_out):
            return None
        names = record.get("artifacts")
        if not isinstance(names, list) or not names:
            return None
        for name in names:
            if not isinstance(name, str) or Path(name).name != name:
                return None
            artifact = Path(channel_out) / name
            try:
                if not artifact.is_file() or artifact.stat().st_size <= 0:
                    return None
            except OSError:
                return None
        return tuple(names)

    def read_claim(self, channel_out: Path, video_id: str) -> Optional[ClaimInfo]:
        path = self.claim_path(channel_out, video_id) / "state.json"
        try:
            with path.open("r", encoding="utf-8") as source:
                state = json.load(source)
            age = max(0.0, time.time() - path.stat().st_mtime)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        return ClaimInfo(
            video_id=str(state.get("video_id", video_id)),
            owner=str(state.get("owner", "")),
            hostname=str(state.get("hostname", "")),
            pid=int(state.get("pid") or 0),
            age_seconds=age,
            progress=state.get("progress") if isinstance(state.get("progress"), dict) else {},
        )

    def try_acquire(self, channel_out: Path, video_id: str) -> Tuple[Optional[JobLease], Optional[ClaimInfo]]:
        channel_out = Path(channel_out).resolve()
        self.pipeline_id_for(channel_out)
        claim_path = self.claim_path(channel_out, video_id)
        token = uuid.uuid4().hex
        recovered_stale = False
        for _ in range(4):
            try:
                claim_path.mkdir()
            except FileExistsError:
                info = self.read_claim(channel_out, video_id)
                state_path = claim_path / "state.json"
                try:
                    age = max(0.0, time.time() - state_path.stat().st_mtime)
                except FileNotFoundError:
                    try:
                        age = max(0.0, time.time() - claim_path.stat().st_mtime)
                    except FileNotFoundError:
                        continue
                if age <= self.lease_seconds:
                    return None, info
                stale_path = self.root(channel_out) / "stale" / f"{job_key(video_id)}.{uuid.uuid4().hex}"
                try:
                    os.rename(claim_path, stale_path)
                except FileNotFoundError:
                    continue
                except OSError:
                    return None, info
                shutil.rmtree(stale_path, ignore_errors=True)
                recovered_stale = True
                continue

            directory_fd = os.open(claim_path, os.O_RDONLY | os.O_DIRECTORY)
            lease = JobLease(
                self, channel_out, video_id, claim_path, token, directory_fd,
                recovered_stale=recovered_stale,
            )
            try:
                lease._write_state({"status": "claimed"})
                lease.assert_owned()
            except BaseException:
                lease.release()
                raise
            return lease, None
        return None, self.read_claim(channel_out, video_id)
