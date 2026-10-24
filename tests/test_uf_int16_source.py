"""Source byte fusion preserves normalization, border replication and ordering."""
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref

import numpy as np
import torch
import torch.nn.functional as F

from src.int16 import ops
from src.int16.model import pad_right_bottom


class SourceBytesTests(unittest.TestCase):
    @staticmethod
    def reference(values):
        return F.pixel_unshuffle(pad_right_bottom(ops.from_bytes(values), 16), 8)

    def test_cpu_reference_all_byte_values_and_padding(self):
        gen = torch.Generator().manual_seed(2225)
        for shape in ((1, 3, 1, 1), (2, 3, 17, 19), (1, 24, 143, 257), (1, 3, 16, 16)):
            values = torch.randint(256, shape, dtype=torch.uint8, generator=gen)
            self.assertTrue(torch.equal(ops.from_bytes_unshuffle(values), self.reference(values)))


class PreparedLifetimeTests(unittest.TestCase):
    def test_quantization_views_are_reused_without_copying_parameters(self):
        from src.int16.layers import Layers
        values = torch.arange(12, dtype=torch.int16).reshape(4, 3)
        layers = Layers(torch.nn.Identity(), {'tensors': {'q': values}}, 'cpu')
        first = layers.parameter('q', 1)
        self.assertIs(layers.parameter('q', 1), first)
        self.assertTrue(torch.equal(first, values[1:2, :, None, None]))
        self.assertEqual(first.untyped_storage().data_ptr(), values.untyped_storage().data_ptr())
        second = layers.parameter('q', 2)
        self.assertIsNot(second, first)
        self.assertTrue(torch.equal(second, values[2:3, :, None, None]))

    def test_checkpoint_released_before_loading_the_next_network(self):
        from src.int16.codec import IntegerCodec, MAGIC
        for image_only in (False, True):
            weak = []
            calls = []

            def resolve(source, variant, path, identity, source_hash):
                if weak:
                    self.assertIsNone(weak[-1]())
                value = torch.zeros(1, dtype=torch.int16)
                weak.append(weakref.ref(value))
                self.assertEqual(identity, ('11' if variant == 'image' else '22')*32)
                self.assertEqual(source_hash, 'hash-i' if variant == 'image' else 'hash-p')
                return path, {'identity': identity, 'tensors': {'weight': value}}

            class Model:
                def __init__(self, data, device, reconstruct_on_encode=True):
                    calls.append((data['identity'], str(device), reconstruct_on_encode))

            with patch('src.int16.codec.resolve_prepared', side_effect=resolve), \
                 patch('src.int16.codec.IntegerModel', Model):
                codec = IntegerCodec('image', 'video', 'htl', device='cpu',
                    prepared_i='prepared-i', prepared_p='prepared-p',
                    identities=('11'*32, '22'*32), source_hashes=('hash-i', 'hash-p'),
                    skip_threshold=0.2, image_only=image_only)
            self.assertTrue(all(value() is None for value in weak))
            self.assertEqual(len(calls), 1 if image_only else 2)
            self.assertTrue(calls[0][2])
            if not image_only:
                self.assertFalse(calls[1][2])
            self.assertEqual(codec.preamble, MAGIC+bytes.fromhex('11'*32)+
                             (bytes(32) if image_only else bytes.fromhex('22'*32)))
            self.assertEqual(codec.image.skip_threshold, 102)


class CudaSourceBytesTests(SourceBytesTests):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest('CUDA not available')
        import uf_int16_cuda
        if not callable(getattr(uf_int16_cuda, 'from_bytes_unshuffle', None)):
            raise unittest.SkipTest('Rebuild UF integer extension for fused source bytes')
        cls.native = uf_int16_cuda

    def test_codec_maps_checkpoints_and_checks_them_after_device_copy(self):
        from src.int16.codec import IntegerCodec
        from src.int16.prepared import content_hash
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pt'
            data = {'format': 'dcvc-uf-prepared-v1', 'arithmetic': ops.ARITHMETIC_ID,
                    'variant': 'image', 'source_sha256': 'a'*64,
                    'tensors': {'weight': torch.arange(16, dtype=torch.int16)}}
            data['identity'] = content_hash(data)
            for change in (False, True):
                torch.save(data, path)
                class Model:
                    def __init__(self, prepared, device):
                        self.weight = prepared['tensors']['weight'].to(device)
                        if change:
                            with path.open('ab') as stream:
                                stream.write(b'changed after validation')
                with patch('src.int16.codec.IntegerModel', Model):
                    if change:
                        with self.assertRaisesRegex(ValueError, 'changed while loading'):
                            IntegerCodec('absent', 'absent', prepared_i=path, image_only=True)
                    else:
                        codec = IntegerCodec('absent', 'absent', prepared_i=path, image_only=True)
                        self.assertTrue(torch.equal(codec.image.weight.cpu(), data['tensors']['weight']))

    def test_fused_all_bytes_padding_views_and_batches(self):
        gen = torch.Generator().manual_seed(2226)
        cases = [torch.arange(256, dtype=torch.uint8).reshape(1, 1, 16, 16).expand(2, 3, 16, 16).clone()]
        cases += [torch.randint(256, shape, dtype=torch.uint8, generator=gen) for shape in
                  ((1, 3, 1, 1), (2, 3, 17, 19), (1, 24, 143, 257), (1, 24, 144, 256))]
        for source in cases:
            for values in (source, source.transpose(2, 3)):
                with self.subTest(shape=tuple(values.shape), strides=values.stride()):
                    expected = self.reference(values)
                    actual = self.native.from_bytes_unshuffle(values.cuda())
                    self.assertEqual(actual.dtype, torch.int16)
                    self.assertTrue(actual.is_contiguous())
                    self.assertTrue(torch.equal(actual.cpu(), expected))
                    self.assertTrue(torch.equal(ops.from_bytes_unshuffle(values.cuda()).cpu(), expected))

    def test_invalid_inputs_and_older_library_fallback(self):
        values = torch.arange(256, dtype=torch.uint8).reshape(1, 1, 16, 16)
        for source in (values, values.cuda().short(), values.cuda()[0], values.cuda()[:, :, :0]):
            with self.assertRaises(RuntimeError):
                self.native.from_bytes_unshuffle(source)
        with patch.object(self.native, 'from_bytes_unshuffle', None):
            self.assertTrue(torch.equal(ops.from_bytes_unshuffle(values.cuda()).cpu(), self.reference(values)))

    def test_encode_hook_retains_public_tensor_and_saved_model_fallback(self):
        from src.int16.codec import IntegerCodec
        from src.int16.layers import Layers
        from src.int16.model import IntegerModel
        codec = IntegerCodec.__new__(IntegerCodec)
        codec.device = torch.device('cuda:0')
        codec.image = IntegerModel.__new__(IntegerModel)
        codec.image.layers = Layers.__new__(Layers)
        codec.video = None
        frames = [np.arange(3*17*19, dtype=np.uint8).reshape(3, 17, 19)]
        original = codec.tensor(frames)
        self.assertEqual(original.dtype, torch.int16)
        raw = codec.tensor_for_encode(frames)
        self.assertEqual(raw.dtype, torch.uint8)
        self.assertTrue(torch.equal(ops.from_bytes_unshuffle(raw),
                                    F.pixel_unshuffle(pad_right_bottom(original, 16), 8)))
        codec.image = SimpleNamespace()
        self.assertTrue(torch.equal(codec.tensor_for_encode(frames), original))
        codec.image = IntegerModel.__new__(IntegerModel)
        codec.image.layers = Layers.__new__(Layers)
        with patch.object(self.native, 'from_bytes_unshuffle', None):
            self.assertTrue(torch.equal(codec.tensor_for_encode(frames), original))
        codec.device = torch.device('cpu')
        self.assertTrue(torch.equal(codec.tensor_for_encode(frames), original.cpu()))

    def test_fused_yuv420_unshuffle_matches_expanded_yuv444(self):
        from src.int16.codec import IntegerCodec
        from src.int16.layers import Layers
        from src.int16.model import IntegerModel
        gen = np.random.default_rng(2227)
        codec = IntegerCodec.__new__(IntegerCodec)
        codec.device = torch.device('cuda:0')
        codec.image = IntegerModel.__new__(IntegerModel)
        codec.image.layers = Layers.__new__(Layers)
        codec.video = None
        for t, h, w in ((1, 16, 16), (1, 18, 22), (8, 144, 256), (8, 30, 34)):
            raw_frames = [gen.integers(0, 256, size=(h * 3 // 2, w), dtype=np.uint8) for _ in range(t)]
            expanded_frames = [codec._expand_yuv420_frame(f) for f in raw_frames]
            packed = codec.tensor_for_encode(raw_frames)
            expanded = codec.tensor_for_encode(expanded_frames)
            self.assertEqual(packed.dim(), 5)
            self.assertEqual(expanded.dim(), 4)
            self.assertTrue(torch.equal(ops.from_bytes_unshuffle(packed), ops.from_bytes_unshuffle(expanded)))
            self.assertTrue(torch.equal(ops.from_bytes_unshuffle(packed.cpu()), ops.from_bytes_unshuffle(expanded.cpu())))


if __name__ == '__main__':
    unittest.main()
