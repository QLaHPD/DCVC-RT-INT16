"""Reset references reuse the decoded last frame without repeating its head."""
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

from src.int16.model import IntegerModel


class ReconstructionTests(unittest.TestCase):
    def exercise(self, variant, reset, output_pixels, output_frames=None):
        model = IntegerModel.__new__(IntegerModel)
        model.variant = variant
        model.model = SimpleNamespace(decoder=object(), recon_head=object())
        feature = torch.arange(32, dtype=torch.int16).reshape(1, 8, 2, 2)
        references = [torch.arange(3*64*2*3, dtype=torch.int16).reshape(1, 192, 2, 3)+index
                      for index in range(1 if variant == 'ld' else 8)]
        frames = [F.pixel_shuffle(value, 8) for value in references]
        pixels = frames[0] if variant == 'ld' else frames
        calls = []

        class Layers:
            def parameter(self, *args):
                return None

            def __call__(self, module, *args, **kwargs):
                calls.append((module, kwargs))
                if module is model.model.decoder:
                    return feature
                if module is model.model.recon_head:
                    if kwargs.get('for_reset'):
                        return references[-1]
                    return frames[:kwargs['output_frames']] if 'output_frames' in kwargs else pixels
                raise AssertionError('unexpected layer')

        model.layers = Layers()
        model.context = object()
        model.memory = object()
        result = model._reconstruct(None, 14, reset, output_pixels, output_frames=output_frames)
        partial = output_frames is not None and output_frames < len(frames)
        if output_pixels and partial:
            self.assertEqual(len(result), output_frames)
            for actual, expected in zip(result, frames):
                self.assertIs(actual, expected)
        else:
            self.assertIs(result, pixels if output_pixels else None)
        head_calls = [options for module, options in calls if module is model.model.recon_head]
        self.assertEqual(len(head_calls), int(reset or output_pixels)+int(reset and output_pixels and partial))
        if reset:
            self.assertTrue(torch.equal(model.reference, references[-1]))
            self.assertIsNone(model.context)
            self.assertIsNone(model.memory)
            self.assertEqual(bool(head_calls[-1].get('for_reset')), not output_pixels or partial)
        else:
            self.assertIs(model.reference, feature)
            self.assertIsNotNone(model.context)
            self.assertIsNotNone(model.memory)

    def test_all_variants_reset_and_pixel_modes(self):
        for variant in ('htl', 'hts', 'ld'):
            for reset in (False, True):
                for output in (False, True):
                    with self.subTest(variant=variant, reset=reset, output=output):
                        self.exercise(variant, reset, output)

    def test_partial_chunks_keep_the_padded_last_frame_as_reset_reference(self):
        for variant in ('htl', 'hts'):
            for count in range(1, 9):
                for reset in (False, True):
                    with self.subTest(variant=variant, count=count, reset=reset):
                        self.exercise(variant, reset, True, count)

    def test_invalid_output_counts_fail_before_decoder_state_changes(self):
        model = IntegerModel.__new__(IntegerModel)
        for variant, maximum in (('image', 1), ('ld', 1), ('hts', 8), ('htl', 8)):
            model.variant = variant
            for count in ((0,) if variant == 'image' else ()) + (-1, maximum+1, True, 1.5):
                with self.subTest(variant=variant, count=count):
                    with self.assertRaisesRegex(ValueError, 'frame count'):
                        model._decompress(b'', {}, 14, 1, output_frames=count)

    def test_skipped_chunks_keep_features_and_reset_reference_without_output_heads(self):
        for variant in ('htl', 'hts', 'ld'):
            for reset in (False, True):
                with self.subTest(variant=variant, reset=reset):
                    self.exercise(variant, reset, False, 0)

    def test_skipped_decode_hook_checks_output_and_keeps_reference_fallback(self):
        from unittest.mock import Mock
        from src.int16.codec import IntegerCodec
        codec = IntegerCodec.__new__(IntegerCodec)
        codec.video = IntegerModel.__new__(IntegerModel)
        codec.video.decompress = Mock(return_value={'x_hat': []})
        self.assertTrue(codec.skip_video(b'bits', {}, 14, 1, True))
        self.assertEqual(codec.video.decompress.call_args.kwargs, {'output_frames': 0})
        codec.video.decompress.return_value = {'x_hat': [object()]}
        with self.assertRaisesRegex(ValueError, 'skipped chunk'):
            codec.skip_video(b'bits', {}, 14, 1, True)
        codec.video = SimpleNamespace(decompress=Mock(side_effect=AssertionError('unexpected decode')))
        self.assertFalse(codec.skip_video(b'bits', {}, 14, 1, True))

    def test_archive_hook_and_reference_implementation_fallback(self):
        from unittest.mock import Mock
        from src.int16.codec import IntegerCodec
        from src.int16.layers import Layers
        codec = IntegerCodec.__new__(IntegerCodec)
        codec.chunk_size = 8
        codec.video = IntegerModel.__new__(IntegerModel)
        codec.video.layers = Layers.__new__(Layers)
        codec.video.decompress = Mock(return_value={'x_hat': 'frames'})
        for count in (1, 3, 8, 10):
            result, expected = codec.decode_video(b'bits', {}, 14, 1, True, count)
            self.assertEqual(expected, min(count, 8))
            self.assertEqual(result, {'x_hat': 'frames'})
            self.assertEqual(codec.video.decompress.call_args.kwargs,
                             {'output_frames': count} if count < 8 else {})
        for old_model, old_layers in ((True, False), (False, True)):
            if old_model:
                codec.video = SimpleNamespace(decompress=Mock(return_value={}),
                                              layers=Layers.__new__(Layers))
            else:
                codec.video = IntegerModel.__new__(IntegerModel)
            if old_layers:
                codec.video.layers = SimpleNamespace()
                codec.video.decompress = Mock(return_value={})
            _, expected = codec.decode_video(b'bits', {}, 14, 1, True, 3)
            self.assertEqual(expected, 8)
            self.assertEqual(codec.video.decompress.call_args.kwargs, {})


class PartialHeadTests(unittest.TestCase):
    def test_all_variants_only_run_requested_heads_and_keep_shared_pairs(self):
        from src.int16.layers import Layers

        class ReconHead(torch.nn.Module):
            def __init__(self, variant):
                super().__init__()
                if variant == 'ld':
                    self.head, self.conv = torch.nn.Identity(), torch.nn.Identity()
                else:
                    self.is_hts = variant == 'hts'
                    if self.is_hts:
                        self.conv1 = torch.nn.ModuleList(torch.nn.Identity() for _ in range(4))
                        self.conv2 = torch.nn.ModuleList(torch.nn.Identity() for _ in range(8))
                    else:
                        self.conv = torch.nn.ModuleList(torch.nn.Identity() for _ in range(8))

        class RecordingLayers(Layers):
            def __call__(self, module, *args, **kwargs):
                if isinstance(module, torch.nn.Identity):
                    self.calls.append(module)
                    return torch.full((1, 192, 1, 1), self.labels.get(id(module), 0), dtype=torch.int16)
                return super().__call__(module, *args, **kwargs)

        for variant in ('ld', 'hts', 'htl'):
            head = ReconHead(variant)
            layers = RecordingLayers(head, {'tensors': {}}, 'cpu')
            heads = [head.head] if variant == 'ld' else list(head.conv2 if variant == 'hts' else head.conv)
            layers.labels = {id(child): index+1 for index, child in enumerate(heads)}
            for count in range(1, len(heads)+1):
                layers.calls = []
                result = layers(head, torch.zeros(1, dtype=torch.int16), output_frames=count)
                frames = result if isinstance(result, list) else [result]
                self.assertEqual(len(frames), count)
                self.assertEqual([int(frame.flatten()[0]) for frame in frames], list(range(1, count+1)))
                self.assertEqual([child for child in layers.calls if id(child) in layers.labels], heads[:count])
                if variant == 'hts':
                    self.assertEqual(sum(child in list(head.conv1) for child in layers.calls), (count+1)//2)
                layers.calls = []
                reference = layers(head, torch.zeros(1, dtype=torch.int16), for_reset=True, output_frames=count)
                self.assertEqual(int(reference.flatten()[0]), len(heads))
                self.assertIn(heads[-1], layers.calls)
            for count in (0, -1, len(heads)+1, True, 1.5):
                with self.assertRaisesRegex(ValueError, 'frame count'):
                    layers(head, torch.zeros(1, dtype=torch.int16), output_frames=count)


class RawChunkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest('CUDA not available')
        import uf_int16_cuda
        if not callable(getattr(uf_int16_cuda, 'raw_frames', None)):
            raise unittest.SkipTest('Rebuild UF integer extension for chunk conversion')
        cls.native = uf_int16_cuda

    def test_full_range_cropping_views_and_partial_chunks(self):
        from src.int16.codec import IntegerCodec
        gen = torch.Generator().manual_seed(2221)
        for count in (1, 3, 8):
            cpu = [torch.randint(-32768, 32768, (1, 3, 10, 14), dtype=torch.int16, generator=gen)
                   for _ in range(count)]
            expected = b''.join(IntegerCodec.raw_frame(frame, 10, 8) for frame in cpu)
            for views in (False, True):
                gpu = [frame.transpose(2, 3).contiguous().transpose(2, 3).cuda() if views else frame.cuda()
                       for frame in cpu]
                actual = self.native.raw_frames(gpu, 10, 8).cpu().numpy().tobytes()
                self.assertEqual(actual, expected)
                self.assertEqual(IntegerCodec.raw_frames(gpu, 10, 8), expected)
        # Every INT16 input value occurs in every component, including tails.
        values = torch.arange(-32768, 32768, dtype=torch.int16).reshape(1, 1, 256, 256)
        frames = [values.expand(1, 3, 256, 256).clone(), values.flip(-1).expand(1, 3, 256, 256).clone()]
        self.assertEqual(self.native.raw_frames([f.cuda() for f in frames], 256, 256).cpu().numpy().tobytes(),
                         b''.join(IntegerCodec.raw_frame(f, 256, 256) for f in frames))

    def test_invalid_chunks_and_older_library_fallback(self):
        from unittest.mock import patch
        from src.int16.codec import IntegerCodec
        frame = torch.zeros((1, 3, 8, 8), dtype=torch.int16, device='cuda')
        for frames, width, height in (([], 8, 8), ([frame]*9, 8, 8), ([frame], 7, 8),
                                      ([frame], 10, 8), ([frame.float()], 8, 8),
                                      ([frame, frame[:, :, :6, :]], 8, 6)):
            with self.subTest(count=len(frames), width=width, height=height):
                with self.assertRaises(RuntimeError):
                    self.native.raw_frames(frames, width, height)
        with patch.object(self.native, 'raw_frames', None):
            self.assertEqual(IntegerCodec.raw_frames([frame]*3, 8, 8),
                             IntegerCodec.raw_frame(frame, 8, 8)*3)


if __name__ == '__main__':
    unittest.main()
