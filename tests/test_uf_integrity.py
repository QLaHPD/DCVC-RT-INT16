"""Regression cases for source completeness and crash-safe cooperative work."""
import contextlib
import functools
import http.server
import io
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from src.archive.coordinator import validate_completed
from src.archive.media import probe
from src.archive.srf_batch import completed
from src.archive.storage import (SharedWorkPool, load_bundle, sha256,
                                 validate_completion, validate_video_coverage)
from src.archive.streaming import existing_stream_archives
from src.archive.workflow import encode_job, run_cleanup


class CountingCodec:
    def encode(self, reader, target, qi, qp, reset, intra, limit, progress):
        frames = 0
        while limit is None or frames < limit:
            if reader.read() is None:
                break
            frames += 1
        Path(target).write_bytes(b'neural stage replaced by frame counter')
        return dict(frames=frames, packets=frames, encode_seconds=0)


def config(audio='none'):
    return dict(codec='dcvc-uf', variant='htl', fps=4, resolution=None, qp_i=15,
                qp_p=39, reset_interval=3, intra_period=-1, max_frames=None,
                audio=audio, opus_channels='mono', opus_bitrate='6k', opus_frame_ms=60)


def ffmpeg(*args):
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', *map(str, args)],
                   check=True, capture_output=True)


class MediaIntegrityTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)

    def encode(self, source, cfg, **extra):
        final = self.root / 'out/item.uf.json'
        job = dict(source=str(source), relative=Path(source).name, final=str(final),
                   layout='flat', **extra)
        with patch('src.archive.workflow.make_codec', return_value=CountingCodec()), \
                contextlib.redirect_stdout(io.StringIO()):
            status = encode_job(job, cfg, (), 'cpu', 'integrity-test')
        return status, final

    def test_truncated_local_mp4_never_commits_or_deletes_input(self):
        full = self.root / 'full.mp4'
        ffmpeg('-f', 'lavfi', '-i', 'testsrc2=size=48x32:rate=4:duration=12',
               '-c:v', 'libx264', '-g', '4', '-bf', '0', '-movflags', '+faststart', full)
        partial = self.root / 'partial.mp4'
        raw = full.read_bytes()
        partial.write_bytes(raw[:len(raw)//2])
        self.assertEqual(probe(partial)['duration'], 12)
        before = sha256(partial)
        status, final = self.encode(partial, config())
        self.assertEqual(status, 'failed')
        self.assertFalse(final.exists())
        self.assertEqual(sha256(partial), before)

    def test_orphan_audio_is_not_reused_or_overwritten(self):
        source = self.root / 'full.mkv'
        ffmpeg('-f', 'lavfi', '-i', 'testsrc2=size=48x32:rate=4:duration=8',
               '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000:duration=8',
               '-c:v', 'ffv1', '-c:a', 'pcm_s16le', source)
        output = self.root / 'out'
        output.mkdir()
        orphan = output / 'item.opus'
        ffmpeg('-f', 'lavfi', '-i', 'sine=frequency=900:sample_rate=48000:duration=0.5',
               '-c:a', 'libopus', '-ac', '1', '-b:a', '6k', orphan)
        before = sha256(orphan)
        status, final = self.encode(source, config('opus'))
        self.assertEqual(status, 'failed')
        self.assertFalse(final.exists())
        self.assertEqual(sha256(orphan), before)
        orphan.unlink()  # Only this test's synthetic uncommitted audio.
        self.assertEqual(self.encode(source, config('opus'))[0], 'encoded')
        data = load_bundle(final)[1]
        self.assertAlmostEqual(data['audio']['duration'], 8, delta=.1)

    def test_long_audio_does_not_make_complete_local_video_look_truncated(self):
        source = self.root / 'long-audio.mkv'
        ffmpeg('-f', 'lavfi', '-i', 'testsrc2=size=48x32:rate=4:duration=4',
               '-f', 'lavfi', '-i', 'sine=duration=8', '-c:v', 'ffv1', '-c:a', 'pcm_s16le', source)
        self.assertEqual(probe(source)['video_duration'], 4)
        self.assertEqual(self.encode(source, config('opus'))[0], 'encoded')

    def test_short_completion_is_rejected_by_all_resume_and_cleanup_gates(self):
        source = self.root / 'full.mkv'
        ffmpeg('-f', 'lavfi', '-i', 'testsrc2=size=48x32:rate=4:duration=5', '-c:v', 'ffv1', source)
        cfg = config()
        status, final = self.encode(source, cfg)
        self.assertEqual(status, 'encoded')
        data = json.loads(final.read_text())
        data['source']['media'].update(duration=12, video_duration=12)
        data['source']['url'] = 'https://example.test/video'
        final.write_text(json.dumps(data))
        for full in (False, True):
            with self.assertRaisesRegex(ValueError, 'ended early'):
                load_bundle(final, cfg, verify_hashes=full)
        job = dict(source=str(source), final=str(final), remote={'url': data['source']['url']})
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(validate_completed(job, cfg), 'failed')
        with self.assertRaisesRegex(ValueError, 'ended early'):
            completed(final, {'url': data['source']['url']}, cfg)
        urls = set()
        existing_stream_archives(final.parent, urls, cfg)
        self.assertEqual(urls, set())
        from types import SimpleNamespace
        with self.assertRaisesRegex(ValueError, 'ended early'):
            run_cleanup(SimpleNamespace(input=str(final), yes=True, base_root=None))
        self.assertTrue(source.exists())

    def test_intentional_partial_and_timing_tolerances(self):
        validate_video_coverage({'frames': 20, 'fps': 4}, {'duration': 12}, dict(config(), max_frames=20))
        validate_video_coverage({'frames': 47, 'fps': 4}, {'duration': 12}, config())
        with self.assertRaisesRegex(ValueError, 'ended early'):
            validate_video_coverage({'frames': 20, 'fps': 4}, {'duration': 12}, config())

    def test_recorded_short_audio_is_rejected(self):
        data = dict(pipeline=config('opus'), source={'media': {'duration': 12, 'audio': True}},
                    video={'frames': 48, 'fps': 4}, artifacts={'audio.opus': {}},
                    audio={'duration': 5, 'expected_duration': 12})
        with self.assertRaisesRegex(ValueError, 'Audio ended early'):
            validate_completion(data)


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@unittest.skipUnless(shutil.which('yt-dlp'), 'Needs yt-dlp in the selected environment')
class HlsIntegrityTests(unittest.TestCase):
    setUp = MediaIntegrityTests.setUp
    def test_missing_middle_fragment_fails_both_download_paths(self):
        from src.archive.stream_pipeline import PreparedDownload, open_descriptor, close_descriptor
        full = self.root / 'full'
        full.mkdir()
        ffmpeg('-f', 'lavfi', '-i', 'testsrc2=size=48x32:rate=4:duration=9',
               '-c:v', 'libx264', '-g', '12', '-bf', '0', '-sc_threshold', '0',
               '-f', 'hls', '-hls_time', '3', '-hls_list_size', '0', full / 'index.m3u8')
        loss = self.root / 'loss'
        shutil.copytree(full, loss)
        (loss / 'index1.ts').unlink()
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0),
            functools.partial(QuietHandler, directory=str(self.root)))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for name in ('full', 'loss'):
                for prepared in (False, True):
                    with self.subTest(source=name, prepared=prepared):
                        remote = dict(url=f'http://127.0.0.1:{server.server_port}/{name}/index.m3u8',
                            binary=shutil.which('yt-dlp'), video_format='best', provider='srf',
                            media=dict(width=48, height=32, fps=4, duration=9, audio=False),
                            public_metadata={'id': 'hls-test'})
                        download = descriptor = None
                        extra = {}
                        try:
                            if prepared:
                                download = PreparedDownload(remote, multiprocessing.get_context('spawn'))
                                descriptor = download.descriptor()
                                open_descriptor(descriptor)
                                extra['prepared_download'] = descriptor
                            # Each case has its own destination; completed peers are never overwritten.
                            final = self.root / f'out-{name}-{prepared}' / 'item.uf.json'
                            job = dict(source='item', relative='item', final=str(final), layout='flat', remote=remote, **extra)
                            with patch('src.archive.workflow.make_codec', return_value=CountingCodec()), \
                                    contextlib.redirect_stdout(io.StringIO()):
                                status = encode_job(job, config(), (), 'cpu', 'hls-test')
                            self.assertEqual(status, 'encoded' if name == 'full' else 'failed')
                            self.assertEqual(final.exists(), name == 'full')
                        finally:
                            if descriptor is not None:
                                close_descriptor(descriptor)
                            if download is not None:
                                download.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


def profile_crash(root, after_publish):
    real_link = os.link
    def crash(*args, **kwargs):
        if after_publish:
            real_link(*args, **kwargs)
        os._exit(23)
    with patch('os.link', side_effect=crash):
        SharedWorkPool().ensure_pipeline(Path(root), {'codec': 'test'})


class CooperativeIntegrityTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_profile_creation_crash_before_and_after_publication(self):
        for after in (False, True):
            root = self.root / str(after)
            process = multiprocessing.get_context('fork').Process(target=profile_crash, args=(str(root), after))
            process.start()
            process.join(10)
            self.assertEqual(process.exitcode, 23)
            profile = SharedWorkPool.root(root) / 'pipeline.json'
            self.assertEqual(profile.exists(), after)
            SharedWorkPool().ensure_pipeline(root, {'codec': 'test'})
            with self.assertRaisesRegex(RuntimeError, 'settings differ'):
                SharedWorkPool().ensure_pipeline(root, {'codec': 'other'})

    def test_delayed_stale_recoverer_cannot_remove_a_fresh_generation(self):
        pool = SharedWorkPool(lease_seconds=30)
        pool.ensure_pipeline(self.root, {'codec': 'test'})
        old, _ = pool.try_acquire(self.root, 'item')
        self.addCleanup(old.release)
        stamp = time.time() - 60
        os.utime(old.claim_path / 'state.json', (stamp, stamp))
        old_inode = os.fstat(old._directory_fd).st_ino
        barrier, acquired = threading.Barrier(2), threading.Event()
        real_listdir = os.listdir
        results, errors = {}, []
        def delayed(fd):
            if isinstance(fd, int) and os.fstat(fd).st_ino == old_inode:
                barrier.wait(5)
                if threading.current_thread().name == 'second':
                    if not acquired.wait(5):
                        raise RuntimeError('First recovery did not complete')
            return real_listdir(fd)
        def recover(label):
            try:
                peer = SharedWorkPool(lease_seconds=30, instance_name=label)
                peer.ensure_pipeline(self.root, {'codec': 'test'})
                results[label] = peer.try_acquire(self.root, 'item')[0]
            except Exception as exc:
                errors.append(exc)
            finally:
                if label == 'first':
                    acquired.set()
        with patch('os.listdir', side_effect=delayed):
            threads = [threading.Thread(target=recover, args=(name,), name=name) for name in ('first', 'second')]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(10)
            self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertIsNotNone(results['first'])
        self.addCleanup(results['first'].release)
        self.assertIsNone(results['second'])
        results['first'].heartbeat()
        self.assertFalse(old._owns_path())


if __name__ == '__main__':
    unittest.main()
