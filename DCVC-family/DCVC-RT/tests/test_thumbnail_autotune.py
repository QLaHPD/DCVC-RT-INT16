import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from PIL import Image
import torch

from src.cli import thumbnail_codec as thumbnails
from src.layers import int16_inference as integer


class ThumbnailAutotuneTests(unittest.TestCase):
    def test_frozen_scope_reuses_cache_without_caching_unseen_fallback(self):
        x = SimpleNamespace(device='cuda:0', shape=(1, 32, 8, 8))
        weight = SimpleNamespace(shape=(32, 32, 1, 1))
        key = (x.device, x.shape, weight.shape, False)
        with mock.patch.object(integer, '_RESIDUAL_TILES', {key: 16}) as cache, \
                mock.patch.object(torch.cuda, 'device', side_effect=AssertionError('unexpected tuning')):
            with integer.residual_autotune_scope(False):
                self.assertEqual(integer._residual_tile(x, weight, None, x), 16)
                x.shape = (1, 32, 16, 16)
                self.assertEqual(integer._residual_tile(x, weight, None, x), 32)
            self.assertEqual(cache, {key: 16})
        self.assertTrue(integer._ALLOW_RESIDUAL_AUTOTUNE.get())

    def test_nested_scope_and_failure_restore_video_tuning(self):
        with self.assertRaisesRegex(RuntimeError, 'failure'):
            with integer.residual_autotune_scope(False):
                with integer.residual_autotune_scope(True):
                    self.assertFalse(integer._ALLOW_RESIDUAL_AUTOTUNE.get())
                    raise RuntimeError('failure')
        self.assertTrue(integer._ALLOW_RESIDUAL_AUTOTUNE.get())

    def test_only_first_image_and_verification_can_tune_per_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, size in [('a.png', (16, 16)), ('b.png', (30, 20)), ('c.png', (16, 16))]:
                Image.new('RGB', size).save(root / name)
            tasks = thumbnails.discover_thumbnails('channel', root, root, qp=45).pending_tasks
            observed = []

            def serialize(*_args):
                observed.append(('encode', integer._ALLOW_RESIDUAL_AUTOTUNE.get()))
                return b'test'

            def verify(*_args):
                observed.append(('verify', integer._ALLOW_RESIDUAL_AUTOTUNE.get()))

            model = SimpleNamespace(parameters=lambda: iter([torch.empty(0)]))
            with mock.patch.object(thumbnails, '_load_image_tensor', return_value=None), \
                    mock.patch.object(thumbnails, '_serialize_intra', side_effect=serialize), \
                    mock.patch.object(thumbnails, '_round_trip_verify', side_effect=verify):
                for task in tasks:
                    thumbnails.encode_thumbnail(task, model)
                another_model = SimpleNamespace(parameters=model.parameters)
                thumbnails.encode_thumbnail(tasks[0], another_model)
            self.assertEqual(observed, [('encode', True), ('verify', True),
                                        ('encode', False), ('verify', False),
                                        ('encode', False), ('verify', False),
                                        ('encode', True), ('verify', True)])
            self.assertTrue(integer._ALLOW_RESIDUAL_AUTOTUNE.get())

    def test_failed_encode_consumes_first_attempt_but_failed_load_does_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new('RGB', (16, 16)).save(root / 'a.png')
            task = thumbnails.discover_thumbnails('channel', root, root, qp=45).pending_tasks[0]
            model = SimpleNamespace(parameters=lambda: iter([torch.empty(0)]))
            with mock.patch.object(thumbnails, '_load_image_tensor', side_effect=RuntimeError('load')):
                with self.assertRaisesRegex(RuntimeError, 'load'):
                    thumbnails.encode_thumbnail(task, model)
            self.assertFalse(getattr(model, '_thumbnail_autotune_started', False))
            seen = []

            def fail(*_args):
                seen.append(integer._ALLOW_RESIDUAL_AUTOTUNE.get())
                raise RuntimeError('encode')

            with mock.patch.object(thumbnails, '_load_image_tensor', return_value=None), \
                    mock.patch.object(thumbnails, '_serialize_intra', side_effect=fail):
                for _ in range(2):
                    with self.assertRaisesRegex(RuntimeError, 'encode'):
                        thumbnails.encode_thumbnail(task, model)
            self.assertEqual(seen, [True, False])
            self.assertTrue(integer._ALLOW_RESIDUAL_AUTOTUNE.get())
            self.assertFalse(Path(task.output_path).exists())


if __name__ == '__main__':
    unittest.main()
