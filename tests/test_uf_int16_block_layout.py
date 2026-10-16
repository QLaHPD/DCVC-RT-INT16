"""Channel-last block intermediates retain every NCHW arithmetic boundary."""
import unittest

import torch

from src.int16 import ops


class BlockLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest('CUDA not available')
        import uf_int16_cuda
        if not getattr(uf_int16_cuda, 'block_channels_last', False):
            raise unittest.SkipTest('Rebuild UF integer extension for block layouts')
        cls.native = uf_int16_cuda

    @staticmethod
    def reference(x, weights, biases, lut, shortcut, cpu=True):
        conv = ops.conv2d_reference if cpu else ops.conv2d
        first = ops.lookup(conv(x, weights[0], biases[0]), lut)
        spatial = conv(first, weights[1], biases[1], padding=(1, 1), groups=weights[1].shape[0])
        residual = ops.add(conv(spatial, weights[2], biases[2]), x)
        expanded = ops.wsilu4(conv(residual, weights[3], biases[3]), lut)
        result = ops.add(conv(expanded, weights[4], biases[4]), residual)
        return ops.add(result, x) if shortcut else result

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

    def test_independent_cpu_both_ratios_batches_tails_and_saturation(self):
        gen = torch.Generator().manual_seed(2222)
        cases = [((1, 384, 9, 16), 1), ((2, 416, 11, 13), 1),
                 ((1, 512, 9, 16), 1), ((1, 512, 9, 16), 2)]
        for shape, ratio in cases:
            for bound in (90, 32768):
                x, weights, biases, lut = self.inputs(shape, ratio, bound, gen)
                args = (x.cuda(), [w.cuda() for w in weights],
                        [b.cuda() if b is not None else None for b in biases], lut.cuda())
                for shortcut in (False, True):
                    with self.subTest(shape=shape, ratio=ratio, bound=bound, shortcut=shortcut):
                        expected = self.reference(x, weights, biases, lut, shortcut)
                        actual = self.native.depth_conv_block(*args, shortcut)
                        self.assertTrue(actual.is_contiguous())
                        self.assertTrue(torch.equal(expected, actual.cpu()))

    def test_large_spatial_tiles_and_older_layout_fallbacks(self):
        gen = torch.Generator().manual_seed(2223)
        for shape, ratio in [((1, 512, 1, 129), 1), ((1, 512, 18, 32), 2),
                             ((1, 512, 33, 33), 1), ((1, 768, 9, 16), 2)]:
            x, weights, biases, lut = self.inputs(shape, ratio, 90, gen)
            gx, gw, gb, gl = x.cuda(), [w.cuda() for w in weights], [
                b.cuda() if b is not None else None for b in biases], lut.cuda()
            expected = self.reference(gx, gw, gb, gl, True, cpu=False)
            actual = self.native.depth_conv_block(gx, gw, gb, gl, True)
            self.assertTrue(torch.equal(expected, actual), str((shape, ratio)))
        # A two-byte storage offset and noncontiguous spatial views retain the
        # existing staged implementation instead of imposing new alignment.
        x, weights, biases, lut = self.inputs((1, 512, 9, 16), 1, 90, gen)
        gx = torch.empty(x.numel()+1, dtype=torch.int16, device='cuda')[1:].reshape(x.shape).copy_(x)
        gw = [w.cuda() for w in weights]; gb = [b.cuda() if b is not None else None for b in biases]
        gl = lut.cuda()
        self.assertEqual(gx.data_ptr()%4, 2)
        for value in (gx, gx.transpose(-1, -2)):
            self.assertTrue(torch.equal(self.reference(value, gw, gb, gl, True, cpu=False),
                                        self.native.depth_conv_block(value, gw, gb, gl, True)))


if __name__ == '__main__':
    unittest.main()
