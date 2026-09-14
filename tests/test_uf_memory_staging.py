from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_uf_archive import ArchiveTests as Fixtures, FakeCodec
from src.archive.storage import completed_disk_stage, memory_stage_root, SharedWorkPool, load_bundle


class MemoryStagingTests(unittest.TestCase):
    setUp = Fixtures.setUp
    encode = Fixtures.encode

    def test_encode_has_no_output_stage_until_finished(self):
        seen = []
        original = FakeCodec.encode
        def encode(codec, reader, path, *args):
            seen.append(Path(path).parent)
            self.assertEqual(Path(path).parent.parent, memory_stage_root())
            self.assertFalse(list(self.final.parent.glob('.*.stage-*')))
            self.assertFalse(self.final.exists())
            return original(codec, reader, path, *args)
        with patch.object(FakeCodec, 'encode', encode):
            self.assertEqual(self.encode(), 'encoded')
        self.assertFalse(seen[0].exists())
        load_bundle(self.final)

    def test_failed_encode_removes_ram_and_creates_no_disk_payload(self):
        seen = []
        def fail(codec, reader, path, *args):
            seen.append(Path(path).parent)
            Path(path).write_bytes(b'unfinished')
            raise OSError('test failure')
        with patch.object(FakeCodec, 'encode', fail):
            self.assertEqual(self.encode(), 'failed')
        self.assertFalse(seen[0].exists())
        self.assertFalse(list(self.final.parent.glob('.*.stage-*')))
        self.assertFalse(self.final.exists())

    def test_publication_copy_failure_cleans_disk_stage(self):
        self.final.parent.mkdir()
        with tempfile.TemporaryDirectory(dir=memory_stage_root()) as ram:
            (Path(ram)/'video.bin').write_bytes(b'complete')
            with patch('src.archive.storage.os.fsync', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    completed_disk_stage(ram, self.final, SimpleNamespace(check=lambda: None))
        self.assertFalse(list(self.final.parent.glob('.*.stage-*')))

    def test_missing_tmpfs_never_falls_back_to_disk(self):
        with patch.object(Path, 'read_text', return_value='/dev/sda / ext4 rw 0 0\n'):
            with self.assertRaisesRegex(RuntimeError, 'tmpfs'):
                memory_stage_root()

    def test_pipeline_error_explains_legacy_opus_default(self):
        pool = SharedWorkPool(instance_name='test')
        pool.ensure_pipeline(self.root, self.config)
        with self.assertRaisesRegex(RuntimeError, 'opus_frame_ms: saved=20, requested=60'):
            SharedWorkPool(instance_name='other').ensure_pipeline(self.root, dict(self.config, opus_frame_ms=60))


if __name__ == '__main__':
    unittest.main()
