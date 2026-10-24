"""Coordinate GPU work between INT16 HT-L workers in one run."""

from collections import Counter
import fcntl
from pathlib import Path
import tempfile


class GpuGate:
    """An advisory lock automatically released if its owning process exits."""
    def __init__(self, path):
        self.path = str(path)
        self.stream = None
        self._fd = None

    def _ensure_open(self):
        if self.stream is None or self.stream.closed:
            self.stream = open(self.path, 'ab')
            self._fd = self.stream.fileno()

    def close(self):
        if self.stream is not None:
            try:
                if not self.stream.closed:
                    self.stream.close()
            finally:
                self.stream = None
                self._fd = None

    def __del__(self):
        self.close()

    def __enter__(self):
        self._ensure_open()
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX)
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *exc):
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)


class EncodeGates:
    """Coordinate HT-L workers; decode jobs resolve their mode per archive."""
    def __init__(self, config, devices):
        self.temporary = None
        self.paths = {}
        if config is not None and (config.get('runtime') != 'int16' or config.get('variant') != 'htl'):
            return
        shared = [device for device, count in Counter(devices).items()
                  if device != 'cpu' and count > 1]
        if shared:
            self.temporary = tempfile.TemporaryDirectory(prefix='uf-gpu-')
            self.paths = {device: str(Path(self.temporary.name)/f'device-{index}.lock')
                          for index, device in enumerate(shared)}

    def job(self, job, device):
        path = self.paths.get(device)
        return {**job, 'gpu_gate': path} if path else job

    def close(self):
        if self.temporary is not None:
            self.temporary.cleanup()
