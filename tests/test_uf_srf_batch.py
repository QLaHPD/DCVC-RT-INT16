import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import shlex
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.archive.media import FrameReader, dimensions, probe
from src.archive.srf_batch import (SHOW, ReferenceSink, archive_path, build_job,
                                   codec_options, read_inventory, run_locked)
from src.archive.storage import load_bundle
from src.archive.streaming import download_pipe
from src.archive.workflow import encode_job, run_cleanup


class ConsumeCodec:
    def encode(self, reader, path, qi, qp, reset, intra, limit, progress):
        frames = 0
        while reader.read() is not None:
            frames += 1
        Path(path).write_bytes(b'encoded')
        return dict(frames=frames, encode_seconds=1, packets=frames)


class SrfTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'input.mkv'
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=48x32:rate=8',
                        '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=16000', '-t', '1',
                        '-c:v', 'ffv1', '-c:a', 'pcm_s16le', str(self.source)], check=True)
        self.row = dict(id='63fe2a20-c70f-4160-bf9e-f4a330d91dfd', provider='srf', show_id=SHOW,
                        timestamp=1792344600, duration_seconds=1, title='Full broadcast',
                        url='https://www.srf.ch/play/tv/popupvideoplayer?id=63fe2a20-c70f-4160-bf9e-f4a330d91dfd',
                        extractor_url='srgssr:srf:video:63fe2a20-c70f-4160-bf9e-f4a330d91dfd')
        self.config = dict(codec='dcvc-uf', variant='hts', fps=8, resolution=32,
                           qp_i=15, qp_p=39, reset_interval=3, intra_period=-1,
                           max_frames=None, audio='opus', opus_channels='mono', opus_bitrate='6k')
        self.info = dict(id=self.row['id'], duration=1, formats=[
            dict(format_id='hls360', url='unused', width=48, height=32, fps=8,
                 vcodec='h264', acodec='aac', tbr=100),
            dict(format_id='hls720', url='unused', width=960, height=640, fps=8,
                 vcodec='h264', acodec='aac', tbr=3000)])
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)

    def job(self):
        with patch('src.archive.srf_batch.metadata', return_value=self.info):
            return build_job(self.row, archive_path(self.root / 'output', self.row), 'yt-dlp', self.config, 0)

    def encode(self, job):
        @contextlib.contextmanager
        def download(*_args, **_kwargs):
            with self.source.open('rb') as stream:
                yield stream
        with patch('src.archive.streaming.download_pipe', side_effect=download) as calls, \
                patch('src.archive.workflow.make_codec', return_value=ConsumeCodec()), \
                contextlib.redirect_stdout(io.StringIO()):
            result = encode_job(job, self.config, ('unused', 'unused'), 'cpu', 'test')
        return result, calls.call_count

    def test_single_download_has_all_video_and_audio_and_reference(self):
        job = self.job()
        reference = self.root / 'reference.yuv'
        job['reference_output'] = str(reference)
        status, calls = self.encode(job)
        self.assertEqual((status, calls), ('encoded', 1))
        root, data = load_bundle(job['final'])
        self.assertEqual(data['source']['kind'], 'srf')
        self.assertEqual(data['video']['frames'], 8)
        self.assertIn('audio.opus', data['artifacts'])
        self.assertEqual(reference.stat().st_size, 48 * 32 * 3 // 2 * 8)
        self.assertTrue(root.is_dir())
        self.assertEqual(self.encode(job), ('resumed', 0))

    def test_actual_space_guard_never_publishes(self):
        job = self.job()
        job['min_free_bytes'] = 10 ** 20
        self.assertEqual(self.encode(job)[0], 'failed')
        self.assertFalse(Path(job['final']).exists())
        self.assertFalse(list(Path(job['final']).parent.glob('*.bin')))

    def test_streamed_archive_cannot_delete_a_local_original(self):
        job = self.job()
        self.assertEqual(self.encode(job)[0], 'encoded')
        with self.assertRaisesRegex(ValueError, 'no stored original'):
            run_cleanup(SimpleNamespace(input=job['final'], base_root=None, yes=True))

    def test_resume_does_not_resolve_or_start_gpu(self):
        job = self.job()
        self.assertEqual(self.encode(job)[0], 'encoded')
        inventory = self.root / 'inventory.jsonl'
        inventory.write_text(json.dumps(self.row) + '\n')
        options = SimpleNamespace(target_psnr_selection={})
        args = SimpleNamespace(inventory=str(inventory), ids=None, validate_pilot=False,
                               ytdlp_bin='ffmpeg', min_free_gib=0, limit=None,
                               max_consecutive_failures=20, cuda_idx=0)
        output = self.root / 'output'
        with patch('src.archive.srf_batch.codec_options', return_value=options), \
                patch('src.archive.srf_batch.pipeline', return_value=(self.config, ('unused', 'unused'))), \
                patch('src.archive.srf_batch.metadata', side_effect=AssertionError('network on resume')), \
                patch('src.archive.srf_batch.PersistentEncoder', side_effect=AssertionError('GPU on resume')), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_locked(args, output), 0)
        state = json.loads((output / 'batch-state.json').read_text())
        self.assertEqual(state['counts'], {'resumed': 1})

    def test_selects_best_available_muxed_stream(self):
        remote = self.job()['remote']
        self.assertEqual(remote['video_format'], 'hls720')
        self.assertEqual(remote['audio_format'], 'hls720')
        self.assertTrue(remote['single_pass_audio'])

    def test_low_space_pauses_before_network_or_gpu(self):
        inventory = self.root / 'inventory.jsonl'
        inventory.write_text(json.dumps(self.row) + '\n')
        output = self.root / 'output'
        output.mkdir()
        options = SimpleNamespace(target_psnr_selection={})
        args = SimpleNamespace(inventory=str(inventory), ids=None, validate_pilot=False,
                               ytdlp_bin='ffmpeg', min_free_gib=10 ** 10, limit=None,
                               max_consecutive_failures=20, cuda_idx=0)
        with patch('src.archive.srf_batch.codec_options', return_value=options), \
                patch('src.archive.srf_batch.pipeline', return_value=(self.config, ('unused', 'unused'))), \
                patch('src.archive.srf_batch.metadata', side_effect=AssertionError('network when disk full')), \
                patch('src.archive.srf_batch.PersistentEncoder', side_effect=AssertionError('GPU when disk full')), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_locked(args, output), 75)
        state = json.loads((output / 'batch-state.json').read_text())
        self.assertEqual(state['status'], 'paused_disk_space')
        self.assertFalse((output / 'batch-ledger.jsonl').exists())

    def test_failed_download_cleans_temporary_hls_fragments(self):
        marker = self.root / 'download-directory.txt'
        binary = self.root / 'downloader'
        binary.write_text('#!/bin/sh\npwd > ' + shlex.quote(str(marker))
                          + '\nprintf fragment > ./--Frag1\nprintf bytes\nexit 1\n')
        binary.chmod(0o700)
        with self.assertRaisesRegex(RuntimeError, 'stream failed'):
            with download_pipe(dict(provider='srf', binary=str(binary), url='unused'), 'muxed') as stream:
                self.assertEqual(stream.read(), b'bytes')
        self.assertFalse(Path(marker.read_text().strip()).exists())

    def test_rejects_mixed_show_and_duplicate_inventory(self):
        inventory = self.root / 'inventory.jsonl'
        inventory.write_text(json.dumps(dict(self.row, show_id='compact-show')) + '\n')
        with self.assertRaisesRegex(ValueError, 'another show'):
            read_inventory(inventory)
        inventory.write_text((json.dumps(self.row) + '\n') * 2)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            read_inventory(inventory)

    def test_requested_short_edge_fps_and_preset(self):
        options = codec_options(SimpleNamespace(target_psnr=35, fps=8, resolution=288))
        self.assertEqual((options.runtime, options.fps, options.resolution), ('int16', 8, 288))
        self.assertGreaterEqual(options.target_psnr_selection['measured_psnr'], 35)
        self.assertEqual(dimensions(1920, 1080, 288), (512, 288))
        self.assertEqual(dimensions(640, 480, 288), (384, 288))
        self.assertEqual(dimensions(720, 1280, 288), (288, 512))

    def test_bounded_reference_metric_and_frame_counts(self):
        raw = bytes(range(24))
        sink = ReferenceSink(io.BytesIO(raw * 2), 4, 4, 2)
        sink.write(raw)
        with self.assertRaisesRegex(ValueError, 'counts differ'):
            sink.result()
        sink.write(raw)
        self.assertAlmostEqual(sink.result()['psnr_db'], 99.9)
        chunked = ReferenceSink(io.BytesIO(raw * 9), 4, 4, 9)
        chunked.write(raw)
        chunked.write(raw * 8)
        self.assertAlmostEqual(chunked.result()['psnr_db'], 99.9)


if __name__ == '__main__':
    unittest.main()
