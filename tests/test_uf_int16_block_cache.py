"""Consecutive depth blocks reuse only a matching packed feature result."""
import unittest
from unittest.mock import patch

import torch

from src.int16 import ops
from src.int16.layers import Layers
from src.layers.layers import DepthConvBlock


class PackedBlockCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest('CUDA not available')
        import uf_int16_cuda
        if not callable(getattr(uf_int16_cuda, 'depth_conv_block_packed', None)):
            raise unittest.SkipTest('Rebuild UF integer extension for packed block results')
        cls.native = uf_int16_cuda

    @staticmethod
    def inputs(shape, ratio, bound, gen):
        c = shape[1]; dc = c//ratio
        sizes = [(dc, c, 1, 1), (dc, 1, 3, 3), (c, dc, 1, 1), (dc*4, c, 1, 1), (c, dc, 1, 1)]
        x = torch.randint(-bound, bound, shape, dtype=torch.int16, generator=gen)
        weights = [torch.randint(-bound, bound, size, dtype=torch.int16, generator=gen) for size in sizes]
        biases = [torch.randint(-bound, bound, (size[0],), dtype=torch.int16, generator=gen)
                  if index % 2 == 0 else None for index, size in enumerate(sizes)]
        lut = torch.randint(-32768, 32768, (65536,), dtype=torch.int16, generator=gen)
        return x, weights, biases, lut

    @staticmethod
    def reference(x, weights, biases, lut):
        first = ops.lookup(ops.conv2d_reference(x, weights[0], biases[0]), lut)
        spatial = ops.conv2d_reference(first, weights[1], biases[1], padding=(1, 1), groups=weights[1].shape[0])
        residual = ops.add(ops.conv2d_reference(spatial, weights[2], biases[2]), x)
        expanded = ops.wsilu4(ops.conv2d_reference(residual, weights[3], biases[3]), lut)
        return ops.add(ops.add(ops.conv2d_reference(expanded, weights[4], biases[4]), residual), x)

    def test_cache_matches_independent_cpu_and_next_block(self):
        gen = torch.Generator().manual_seed(2230)
        for shape, ratio in [((1, 384, 9, 16), 1), ((2, 416, 11, 13), 1),
                             ((1, 512, 9, 16), 1), ((1, 512, 18, 32), 2), ((1, 128, 9, 16), 1)]:
            for bound in (90, 32768):
                x, weights, biases, lut = self.inputs(shape, ratio, bound, gen)
                gx, gw, gb, gl = x.cuda(), [w.cuda() for w in weights], [
                    b.cuda() if b is not None else None for b in biases], lut.cuda()
                actual, packed = self.native.depth_conv_block_packed(gx, gw, gb, gl, True, None, True)
                self.assertTrue(actual.is_contiguous())
                self.assertTrue(torch.equal(actual.cpu(), self.reference(x, weights, biases, lut)))
                if packed is not None:
                    self.assertTrue(packed.is_contiguous())
                    self.assertTrue(torch.equal(packed, actual.permute(0, 2, 3, 1)))
                following, empty = self.native.depth_conv_block_packed(actual, gw, gb, gl, False, packed, False)
                self.assertIsNone(empty)
                self.assertTrue(torch.equal(following, self.native.depth_conv_block(actual, gw, gb, gl, False)))

    def test_invalid_packed_layouts_are_rejected(self):
        gen = torch.Generator().manual_seed(2231)
        x, weights, biases, lut = self.inputs((1, 512, 9, 16), 1, 90, gen)
        x, weights, biases, lut = x.cuda(), [w.cuda() for w in weights], [
            b.cuda() if b is not None else None for b in biases], lut.cuda()
        packed = x.permute(0, 2, 3, 1).contiguous()
        unaligned = torch.empty(packed.numel()+1, dtype=torch.int16, device='cuda')[1:].reshape(packed.shape)
        for invalid in (packed.cpu(), packed.float(), packed[:, :, :-1], unaligned):
            with self.assertRaises(RuntimeError):
                self.native.depth_conv_block_packed(x, weights, biases, lut, False, invalid, False)

    def test_sequence_adaptors_small_shapes_and_older_library_fallback(self):
        gen = torch.Generator().manual_seed(2232)
        for channels, h, w, adaptor in ((512, 9, 16, False), (512, 18, 32, True),
                                        (256, 9, 16, False), (512, 3, 7, False)):
            target = 384 if adaptor else channels
            with torch.device('meta'):
                model = torch.nn.Sequential(DepthConvBlock(channels, channels),
                    DepthConvBlock(channels, target), DepthConvBlock(target, target))
            values = {name: torch.randint(-90, 91, value.shape, dtype=torch.int16, generator=gen)
                      for name, value in model.named_parameters()}
            values['tables.wsilu'] = torch.randint(-32768, 32768, (65536,), dtype=torch.int16, generator=gen)
            current = Layers(model, dict(tensors=values), 'cuda')
            x = torch.randint(-90, 91, (1, channels, h, w), dtype=torch.int16, generator=gen).cuda()
            actual = current(model, x)
            with patch.object(self.native, 'depth_conv_block_packed', None):
                previous = Layers(model, dict(tensors=current.values), 'cuda')
                self.assertFalse(previous._packed_chains)
                self.assertTrue(torch.equal(actual, previous(model, x)))


if __name__ == '__main__':
    unittest.main()
