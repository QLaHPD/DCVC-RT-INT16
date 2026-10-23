import threading
import time
from types import SimpleNamespace
import unittest

from src.archive.view_buffer import BufferedDecodeWorker


def until(predicate, timeout=3):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError('Timed out waiting for decode-ahead')


class Source:
    def __init__(self, frames=1200):
        self.index = SimpleNamespace(frame_count=frames)
        self.calls = []
        self.closed = False
        self.delay = 0
        self.fail_at = None

    def seek(self, index, output_format, cancel):
        self.calls.append(index)
        time.sleep(self.delay)
        if cancel():
            raise InterruptedError('cancelled')
        if index == self.fail_at:
            raise ValueError('broken packet')
        assert output_format == 'yuv420p'
        return SimpleNamespace(index=index, rgb=None, raw=b'yuv')

    def close(self):
        self.closed = True


class BufferTests(unittest.TestCase):
    def worker(self, source=None, capacity=512, history=512):
        source = source or Source()
        worker = BufferedDecodeWorker(source, capacity, history)
        self.addCleanup(worker.close)
        return source, worker

    def test_capacity_is_512_ahead_without_any_fps_dependency(self):
        source, worker = self.worker()
        worker.request(1, 0)
        until(lambda: worker.status(1)['size'] == 512)
        time.sleep(.03)
        self.assertEqual(len(source.calls), 513)  # Current frame plus 512 future frames.
        self.assertEqual(worker.pop_due(1, 0).index, 0)
        self.assertEqual(worker.pop_due(1, 1).index, 1)
        until(lambda: worker.status(1)['size'] == 512)
        self.assertEqual(len(source.calls), 514)

    def test_only_consumes_frames_due_for_presentation(self):
        _, worker = self.worker(capacity=8)
        worker.request(1, 0)
        until(lambda: worker.status(1)['size'] == 8)
        self.assertEqual(worker.pop_due(1, 0).index, 0)
        self.assertIsNone(worker.pop_due(1, 0))
        self.assertEqual(worker.pop_due(1, 3).index, 3)
        self.assertIsNone(worker.pop_due(1, 2))

    def test_seek_discards_inflight_and_buffered_frames(self):
        source, worker = self.worker(capacity=2)
        source.delay = .04
        worker.request(1, 0)
        until(lambda: bool(source.calls))
        worker.request(2, 100)
        until(lambda: worker.status(2)['size'] == 2)
        self.assertIsNone(worker.pop_due(1, 1000))
        self.assertEqual(worker.pop_due(2, 100).index, 100)
        self.assertIsNone(worker.pop_due(2, 99))

    def test_short_stream_marks_eof_and_close_releases_a_full_queue(self):
        source, worker = self.worker(Source(frames=3), capacity=3)
        worker.request(1, 0)
        until(lambda: worker.status(1)['eof'])
        self.assertEqual(worker.status(1)['size'], 2)  # Excludes current frame.
        worker.close()
        worker.close()
        self.assertTrue(source.closed)
        self.assertFalse(worker.thread.is_alive())
        self.assertEqual(worker.status(1)['size'], 0)

    def test_previous_512_are_cached_and_the_513th_uses_a_seek(self):
        source, worker = self.worker(Source(frames=3000))
        worker.request(1, 0)
        until(lambda: worker.status(1)['size'] == 512)
        self.assertEqual(worker.pop_due(1, 512).index, 512)
        until(lambda: worker.status(1)['size'] == 512)
        current = worker.pop_due(1, 600)
        self.assertEqual(current.index, 600)
        current.rgb = b'display-only RGB'
        until(lambda: worker.status(1)['size'] == 512)
        self.assertEqual(worker.status(1)['history'], 512)
        self.assertEqual(len(worker.frames), 1025)
        count = len(source.calls)
        for request_id, index in enumerate(range(599, 87, -1), 2):
            worker.request(request_id, index)
            self.assertEqual(worker.pop_due(request_id, index).index, index)
            self.assertLessEqual(worker.status(request_id)['size'], 512)
            self.assertLessEqual(worker.status(request_id)['history'], 512)
        self.assertEqual(len(source.calls), count, 'Previous must not invoke the decoder')
        self.assertIsNone(current.rgb, 'Retained frames must not keep RGB copies')
        worker.request(1000, 87)
        until(lambda: 87 in worker.frames)
        self.assertEqual(worker.pop_due(1000, 87).index, 87)
        self.assertIn(87, source.calls[count:])

    def test_cached_seek_cannot_return_a_frame_from_an_old_ui_request(self):
        source, worker = self.worker(Source(frames=20), capacity=8, history=8)
        worker.request(1, 0)
        until(lambda: worker.status(1)['size'] == 8)
        worker.pop_due(1, 5)
        until(lambda: worker.status(1)['size'] == 8)
        worker.request(2, 4)
        self.assertIsNone(worker.pop_due(1, 5))
        self.assertEqual(worker.pop_due(2, 4).index, 4)
        worker.request(3, 5)
        self.assertEqual(worker.pop_due(3, 5).index, 5)
        self.assertEqual(source.calls.count(4), 1)
        self.assertEqual(source.calls.count(5), 1)

    def test_decode_error_is_reported_and_new_seek_recovers(self):
        source, worker = self.worker(Source(frames=8), capacity=8)
        source.fail_at = 3
        worker.request(1, 0)
        until(lambda: worker.status(1)['error'])
        self.assertIn('broken packet', worker.status(1)['error'])
        worker.request(2, 5)
        until(lambda: worker.status(2)['eof'])
        self.assertIsNone(worker.status(2)['error'])
        self.assertEqual(worker.pop_due(2, 5).index, 5)

    def test_close_cancels_long_seeks_and_joins_the_decoder(self):
        source = Source()
        entered = threading.Event()
        def long_seek(index, output_format, cancel):
            entered.set()
            while not cancel():
                time.sleep(.005)
            raise InterruptedError('cancelled')
        source.seek = long_seek
        _, worker = self.worker(source)
        worker.request(1, 100)
        self.assertTrue(entered.wait(1))
        worker.close()
        self.assertFalse(worker.thread.is_alive())
        self.assertTrue(source.closed)


if __name__ == '__main__':
    unittest.main()
