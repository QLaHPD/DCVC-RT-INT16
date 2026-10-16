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

    def __enter__(self):
        stream = open(self.path, 'ab')
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        except BaseException:
            stream.close()
            raise
        self.stream = stream
        return self

    def __exit__(self, *exc):
        try:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        finally:
            self.stream.close()
            self.stream = None


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
