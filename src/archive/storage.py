"""Atomic UF archive bundles, inventory hashes and cooperative ownership."""

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
import re
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
    record = {'time': datetime.now(timezone.utc).isoformat(), 'event': name, **fields}
    if os.environ.get('UF_JOB_KEY'):
        record['job_key'] = os.environ['UF_JOB_KEY']
    line = json.dumps(record)
    from src.archive.dashboard import event_queue
    destination = event_queue()
    if destination is None:
        print(line, flush=True)
    else:
        destination.put(record)
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


def manifest_path(path):
    path = Path(path).absolute()
    if path.is_dir():
        return path / 'manifest.json'
    if path.name.endswith('.uf.json'):
        return path
    candidate = path.with_suffix('.uf.json')
    if candidate.exists():
        return candidate
    match = re.fullmatch(r'(.+)_\d+x\d+_qI\d+_qP\d+\.bin', path.name)
    if match:
        candidate = path.parent / (match.group(1) + '.uf.json')
        if candidate.exists():
            return candidate
    return path.parent / 'manifest.json'


def artifact_path(root, metadata, name):
    actual = metadata.get('files', {}).get(name, name)
    if not actual or Path(actual).name != actual or actual in ('.', '..'):
        raise ValueError(f'Unsafe artifact path: {actual}')
    return Path(root) / actual


def control_path(root, metadata, name):
    return Path(root) / (metadata['archive_stem'] + '.' + name if metadata.get('layout') == 'flat' else name)


def archive_location(root, metadata):
    return control_path(root, metadata, 'uf.json') if metadata.get('layout') == 'flat' else Path(root)


def archive_lease(root, metadata):
    if metadata.get('layout') == 'flat':
        return Path(root), metadata.get('lease_key', metadata['archive_stem'] + '.uf')
    return Path(root).parent, Path(root).name


def load_bundle(path, expected=None):
    manifest = manifest_path(path)
    if manifest.is_symlink():
        raise ValueError('Archive manifest must not be a symlink')
    root = manifest.parent
    with manifest.open() as stream:
        data = json.load(stream)
    if data.get('format') != 'dcvc-uf-archive' or data.get('version') != 1:
        raise ValueError(f'Not a supported UF archive: {root}')
    if data.get('layout') == 'flat':
        stem = data.get('archive_stem')
        if not isinstance(stem, str) or Path(stem).name != stem or stem in ('', '.', '..'):
            raise ValueError('Invalid flat archive name')
        if manifest.name != stem + '.uf.json':
            raise ValueError('Flat manifest name does not match archive')
        if set(data.get('files', {})) != set(data.get('artifacts', {})):
            raise ValueError('Flat artifact map is incomplete')
        if len(set(data['files'].values())) != len(data['files']):
            raise ValueError('Flat artifacts must have distinct paths')
    if data.get('layout') == 'flat' and Path(path).suffix == '.bin' and artifact_path(root, data, 'video.bin').name != Path(path).name:
        raise ValueError('Input bitstream does not belong to this manifest')
    if expected is not None and data['pipeline'] != expected:
        raise ValueError(f'Existing archive uses different encoding settings/models: {root}')
    if not data.get('validation', {}).get('decoded_yuv_sha256'):
        raise ValueError(f'Archive has no completed decode validation: {root}')
    for name, info in data['artifacts'].items():
        if Path(name).name != name:
            raise ValueError(f'Unsafe artifact name: {name}')
        artifact = artifact_path(root, data, name)
        if artifact.is_symlink() or not artifact.is_file() or artifact.stat().st_size != info['size'] or sha256(artifact) != info['sha256']:
            raise ValueError(f'Archive artifact missing or changed: {artifact}')
    return root, data


def publish_flat(stage, manifest, metadata, heartbeat, lease_key):
    """Publish payloads without overwriting; the manifest is the commit record.

    An interrupted publication can leave payloads without a manifest. Retrying
    accepts only byte-identical payloads, never replaces a final file, and commits
    only after the current owner has finished full decode validation.
    """
    manifest, stage = Path(manifest), Path(stage)
    stem = manifest.name.removesuffix('.uf.json')
    metadata.update(layout='flat', archive_stem=stem, lease_key=lease_key)
    names = {}
    for name in metadata['artifacts']:
        if name == 'video.bin':
            if metadata.get('filename_style') == 'rt':
                video, config = metadata['video'], metadata['pipeline']
                names[name] = f"{stem}_{video['width']}x{video['height']}_qI{config['qp_i']}_qP{config['qp_p']}.bin"
            else:
                names[name] = stem + '.bin'
        elif name == 'audio.opus':
            names[name] = stem + '.opus'
        elif name == 'source.info.json':
            names[name] = stem + ('.info.json' if metadata.get('filename_style') == 'rt' else '.uf.info.json')
        else:
            names[name] = name
    if len(set(names.values())) != len(names):
        raise ValueError('Sidecar names collide with encoded artifacts')
    metadata['files'] = names
    pending = manifest.parent / ('.' + manifest.name + '.pending')
    heartbeat.check()
    atomic_json(pending, {'stage': stage.name, 'metadata': metadata})
    for name, actual in names.items():
        heartbeat.check()
        target = artifact_path(manifest.parent, metadata, name)
        try:
            os.link(stage / name, target)
        except FileExistsError:
            info = metadata['artifacts'][name]
            if target.is_symlink() or not target.is_file() or target.stat().st_size != info['size'] or sha256(target) != info['sha256']:
                raise FileExistsError(f'Refusing to overwrite different output: {target}')
    sync_directory(manifest.parent)
    atomic_json(stage / 'flat-manifest.json', metadata)
    heartbeat.check()
    os.link(stage / 'flat-manifest.json', manifest)
    sync_directory(manifest.parent)
    pending.unlink(missing_ok=True)
    sync_directory(manifest.parent)


def recover_flat(manifest, expected, heartbeat, lease_key, source, source_sha, url=None):
    """Finish a validated interrupted transaction, preserving exact Opus bytes."""
    manifest = Path(manifest)
    pending = manifest.parent / ('.' + manifest.name + '.pending')
    if not pending.exists():
        return None
    if pending.is_symlink():
        raise ValueError('Unsafe publication journal')
    transaction = json.loads(pending.read_text())
    metadata = transaction['metadata']
    if metadata['pipeline'] != expected:
        raise ValueError('Pending publication has different encoding settings')
    previous = metadata['source']
    if (url and previous.get('url') != url) or (not url and
            (previous['path'] != str(source) or previous['sha256'] != source_sha)):
        raise ValueError('Source differs from pending publication')
    name = transaction['stage']
    if Path(name).name != name or not name.startswith('.' + manifest.name + '.stage-'):
        raise ValueError('Unsafe pending stage path')
    stage = manifest.parent / name
    if stage.is_symlink() or not stage.is_dir():
        raise ValueError('Pending stage is missing; refusing to replace published payloads')
    for logical, info in metadata['artifacts'].items():
        if Path(logical).name != logical:
            raise ValueError('Unsafe staged artifact')
        path = stage / logical
        if path.is_symlink() or not path.is_file() or path.stat().st_size != info['size'] or sha256(path) != info['sha256']:
            raise ValueError('Pending staged artifact has changed')
    publish_flat(stage, manifest, metadata, heartbeat, lease_key)
    import shutil
    shutil.rmtree(stage)
    return metadata


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
