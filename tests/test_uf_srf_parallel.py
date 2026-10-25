"""Exercise the shipped multi-GPU scheduler's per-item failure classification."""
import importlib.util
from pathlib import Path
import time
from types import SimpleNamespace
import unittest

from src.archive.srf_batch import classify_error

spec = importlib.util.spec_from_file_location('uf_srf_parallel', Path(__file__).resolve().parents[1] / 'scripts/run_srf_parallel.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


class SrfParallelTests(unittest.TestCase):
    def test_unavailable_items_do_not_abort_valid_inflight_or_queued_jobs(self):
        settled, aborted = [], []
        class Worker:
            def __init__(self, device):
                self.job = None
                self.connection = SimpleNamespace(poll=lambda: time.monotonic() - self.started > 2)
                self.process = SimpleNamespace(is_alive=lambda: True)
            def assign(self, job):
                self.job, self.started = job, time.monotonic()
            def receive(self):
                self.job = None
                return 'encoded'
            def close(self, abort=False):
                if abort:
                    aborted.append(self.job)
        def resolve(row):
            if 1 <= row['id'] <= 25:
                raise RuntimeError('ERROR: [SRGSSR] No video formats found!')
            return row
        scheduler = controller.Scheduler([{'id': i} for i in range(27)], list(range(5)), resolve, Worker,
            completed=lambda row: False, settle=lambda row, status, **kw: settled.append((row['id'], status)),
            space_ok=lambda row: True, state=lambda *args: None, error_limit=20)
        self.assertEqual(scheduler.run(), 0)
        self.assertEqual(sorted(i for i, status in settled if status == 'encoded'), [0, 26])
        self.assertEqual(sum(status == 'unavailable' for _, status in settled), 25)
        self.assertEqual(aborted, [])

    def test_network_and_encoder_errors_still_count_as_failures(self):
        for message in ('HTTP Error 503', 'CUDA out of memory', 'Connection timed out'):
            self.assertEqual(classify_error(RuntimeError(message)), 'failed')
        for message in ('No video formats found!', 'No downloadable video format', 'GEOBLOCK'):
            self.assertEqual(classify_error(ValueError(message)), 'unavailable')


if __name__ == '__main__':
    unittest.main()
