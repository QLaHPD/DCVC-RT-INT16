import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

from src.archive.media import FrameReader, dimensions, probe
from src.archive.storage import SharedWorkPool, load_bundle, sha256
from src.archive.workflow import discover, encode_job, run_cleanup


class FakeCodec:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, reader, path, qi, qp, reset, intra, limit, progress):
        count = 0
        while limit is None or count < limit:
            if reader.read() is None:
                break
            count += 1
        Path(path).write_bytes(b'fake-codec-payload')
        return {'frames': count, 'packets': count, 'encode_seconds': 1.0}

    def decode(self, path, metadata, **kwargs):
        return {'decoded_frames': metadata['frames'], 'decoded_yuv_sha256': 'fake-digest', 'decode_seconds': 1.0}


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.mkv'
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=48x32:rate=4',
                        '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=16000', '-t', '1',
                        '-c:v', 'ffv1', '-c:a', 'pcm_s16le', str(self.source)], check=True)
        self.final = self.root / 'archive/source.mkv.uf'
        self.job = {'source': str(self.source), 'relative': self.source.name, 'final': str(self.final)}
        self.config = {'codec': 'dcvc-uf', 'variant': 'hts', 'fps': None, 'resolution': None,
                       'qp_i': 36, 'qp_p': 30, 'reset_interval': 32, 'intra_period': -1,
                       'max_frames': None, 'audio': 'opus', 'opus_channels': 'mono', 'opus_bitrate': '6k'}
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        codec = patch.dict('sys.modules', {
            'src.archive.codec': types.SimpleNamespace(Codec=FakeCodec),
            'torch': types.SimpleNamespace(__version__='test', version=types.SimpleNamespace(cuda='test'),
                                           cuda=types.SimpleNamespace(get_device_name=lambda _: 'test GPU'))})
        codec.start()
        self.addCleanup(codec.stop)

    def encode(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return encode_job(self.job, self.config, ('unused-i', 'unused-p'), 0, 'test')

    def test_atomic_archive_audio_resume_and_changed_source(self):
        original = sha256(self.source)
        self.assertEqual(self.encode(), 'encoded')
        _, metadata = load_bundle(self.final, self.config)
        self.assertEqual(metadata['video']['frames'], 4)
        self.assertEqual(metadata['source']['sha256'], original)
        self.assertIn('audio.opus', metadata['artifacts'])
        self.assertEqual(sha256(self.source), original)
        self.assertEqual(self.encode(), 'resumed')
        (self.final / 'video.bin').write_bytes(b'corrupt')
        self.assertEqual(self.encode(), 'failed')
        self.assertEqual((self.final / 'video.bin').read_bytes(), b'corrupt')
        self.assertTrue(self.source.exists())

    def test_failure_never_publishes_or_deletes_input(self):
        with patch.object(FakeCodec, 'decode', side_effect=ValueError('bad stream')):
            self.assertEqual(self.encode(), 'failed')
        self.assertFalse(self.final.exists())
        self.assertFalse(list(self.final.parent.glob('*.stage-*')))
        self.assertTrue(self.source.exists())

    def test_peer_claim_and_pipeline_mismatch(self):
        pool = SharedWorkPool(instance_name='other')
        pool.ensure_pipeline(self.final.parent, self.config)
        lease, _ = pool.try_acquire(self.final.parent, self.final.name)
        try:
            self.assertEqual(self.encode(), 'busy')
            self.assertTrue(self.source.exists())
        finally:
            lease.release()
        self.assertEqual(self.encode(), 'encoded')
        changed = dict(self.config, qp_p=29)
        with self.assertRaisesRegex(Exception, 'settings differ'):
            pool.ensure_pipeline(self.final.parent, changed)

    def test_cleanup_preview_and_exact_path(self):
        self.assertEqual(self.encode(), 'encoded')
        unrelated = self.root / 'unrelated.txt'
        unrelated.write_text('keep')
        args = types.SimpleNamespace(input=str(self.final), base_root=None, yes=False)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_cleanup(args), 0)
        self.assertTrue(self.source.exists())
        args.yes = True
        with patch('src.archive.workflow.run_decode', return_value=0), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_cleanup(args), 0)
        self.assertFalse(self.source.exists())
        self.assertEqual(unrelated.read_text(), 'keep')
        load_bundle(self.final)

    def test_cleanup_rejects_partial_and_missing_audio(self):
        self.config['max_frames'] = 2
        self.assertEqual(self.encode(), 'encoded')
        args = types.SimpleNamespace(input=str(self.final), base_root=None, yes=True)
        with self.assertRaisesRegex(ValueError, 'Partial archives'):
            run_cleanup(args)
        self.assertTrue(self.source.exists())
        manifest = self.final / 'manifest.json'
        data = json.loads(manifest.read_text())
        data['pipeline']['max_frames'] = None
        del data['artifacts']['audio.opus']
        manifest.write_text(json.dumps(data))
        # Use a separate parent to avoid intentionally changing the shared pipeline.
        moved = self.root / 'other' / self.final.name
        moved.parent.mkdir()
        self.final.rename(moved)
        args.input = str(moved)
        with self.assertRaisesRegex(ValueError, 'omits source audio'):
            run_cleanup(args)
        self.assertTrue(self.source.exists())

    def test_folder_discovery_avoids_archives_and_handles_duplicate_stems(self):
        other = self.source.with_suffix('.mp4')
        other.write_bytes(b'input')
        archived = self.root / 'old.uf'
        archived.mkdir()
        (archived / 'reconstruction.mkv').write_bytes(b'not input')
        args = types.SimpleNamespace(input_file=None, base_root=str(self.root), channel_ids=None,
                                     recursive=True, output_root=str(self.root / 'out'))
        jobs = discover(args)
        self.assertEqual(len(jobs), 2)
        self.assertEqual(len({job['final'] for job in jobs}), 2)

    def test_aspect_ratio_and_no_unnecessary_resize(self):
        self.assertEqual(dimensions(1280, 720, 144), (256, 144))
        original = probe(self.source)
        with FrameReader(self.source, 48, 32, original, fps=4) as reader:
            self.assertNotIn('scale=', ' '.join(reader.process.args))
            self.assertEqual(reader.read().shape, (3, 32, 48))


if __name__ == '__main__':
    unittest.main()
