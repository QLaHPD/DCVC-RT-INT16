"""Atomic UF archive bundles, inventory hashes and cooperative ownership."""

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading

# Reuse the tested, filesystem-based lease protocol without importing RT models.
_SHARED_PATH = Path(__file__).resolve().parents[2] / 'DCVC-family/DCVC-RT/src/cli/shared_work.py'
_spec = importlib.util.spec_from_file_location('src.archive._shared_protocol', _SHARED_PATH)
_shared = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _shared
_spec.loader.exec_module(_shared)
class SharedWorkPool(_shared.SharedWorkPool):
    @staticmethod
    def root(channel_out):
        return Path(channel_out) / '.dcvc-uf-shared-work'


sha256 = _shared.file_sha256


def event(name, **fields):
    line = json.dumps({'time': datetime.now(timezone.utc).isoformat(), 'event': name, **fields})
    print(line, flush=True)
    log = os.environ.get('UF_PROGRESS_LOG')
    if log:
        fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, (line + '\n').encode())
        finally:
            os.close(fd)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def identity(path):
    value = Path(path).stat()
    return {'size': value.st_size, 'mtime_ns': value.st_mtime_ns}


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def bundle_path(path):
    path = Path(path).resolve()
    return path if path.is_dir() else path.parent


def load_bundle(path, expected=None):
    root = bundle_path(path)
    with (root / 'manifest.json').open() as stream:
        data = json.load(stream)
    if data.get('format') != 'dcvc-uf-archive' or data.get('version') != 1:
        raise ValueError(f'Not a supported UF archive: {root}')
    if expected is not None and data['pipeline'] != expected:
        raise ValueError(f'Existing archive uses different encoding settings/models: {root}')
    if not data.get('validation', {}).get('decoded_yuv_sha256'):
        raise ValueError(f'Archive has no completed decode validation: {root}')
    for name, info in data['artifacts'].items():
        if Path(name).name != name or (root / name).is_symlink():
            raise ValueError(f'Unsafe artifact path: {name}')
        artifact = root / name
        if not artifact.is_file() or artifact.stat().st_size != info['size'] or sha256(artifact) != info['sha256']:
            raise ValueError(f'Archive artifact missing or changed: {artifact}')
    return root, data


class Heartbeat:
    def __init__(self, lease):
        self.lease, self.stopped, self.error = lease, threading.Event(), None
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stopped.wait(20):
            try:
                with self.lock:
                    self.lease.heartbeat()
            except Exception as exc:
                self.error = exc
                return

    def __enter__(self):
        self.thread.start()
        return self

    def check(self):
        if self.error:
            raise RuntimeError(f'Lost shared-work heartbeat: {self.error}')
        self.lease.assert_owned()

    def __exit__(self, *exc):
        self.stopped.set()
        self.thread.join()
