"""Fused convolution/activation must preserve each original rounding boundary."""

import unittest
from unittest.mock import patch

import torch

from src.int16 import ops


class FusedFfnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest('CUDA not available')
        try:
            import uf_int16_cuda
        except ImportError:
            raise unittest.SkipTest('UF integer extension not built')
        if not hasattr(uf_int16_cuda, 'conv2d_wsilu4'):
            raise unittest.SkipTest('Rebuild UF integer extension for fused FFNs')
        cls.native = uf_int16_cuda

    def test_all_tiles_match_cpu_with_partial_rows_and_spatial_tails(self):
        gen = torch.Generator().manual_seed(917)
        lut = torch.randint(-32768, 32768, (65536,), dtype=torch.int16, generator=gen)
        for shape, co in (((2, 35, 3, 7), 20), ((1, 67, 16, 16), 68),
                          ((1, 129, 1, 1), 4)):
            for bound in (90, 32768):
                x = torch.randint(-bound, bound, shape, dtype=torch.int16, generator=gen)
                w = torch.randint(-bound, bound, (co, shape[1], 1, 1), dtype=torch.int16, generator=gen)
                b = torch.randint(-200, 201, (co,), dtype=torch.int16, generator=gen)
                convolution = ops.conv2d_reference(x, w, b)
                expected = ops.wsilu4(convolution, lut)
                gx, gw, gb, gl = x.cuda(), w.cuda(), b.cuda(), lut.cuda()
                for tile in range(1, 6):
                    with self.subTest(shape=shape, bound=bound, tile=tile):
                        self.assertTrue(torch.equal(convolution, self.native.conv2d_tiled(gx, gw, gb, tile).cpu()))
                        self.assertTrue(torch.equal(expected, self.native.conv2d_wsilu4(gx, gw, gb, gl, tile).cpu()))

    def test_saturations_cancellation_ties_and_wide_accumulation(self):
        lut = torch.arange(-32768, 32768, dtype=torch.int16)
        cases = [(torch.zeros((1, 1, 1, 1), dtype=torch.int16),
                  torch.zeros((4, 1, 1, 1), dtype=torch.int16),
                  torch.tensor([32767, 32767, -32768, -32768], dtype=torch.int16)),
                 (torch.full((1, 1, 1, 1), 4096, dtype=torch.int16),
                  torch.tensor([1, -1, 3, -3], dtype=torch.int16).reshape(4, 1, 1, 1), None)]
        x = torch.full((1, 32768, 1, 1), 32767, dtype=torch.int16)
        w = torch.full((4, 32768, 1, 1), 32767, dtype=torch.int16)
        w[1].fill_(-32768)
        w[2, :16384].fill_(-32767)
        w[3].zero_()
        cases.append((x, w, None))
        for x, w, b in cases:
            expected = ops.wsilu4(ops.conv2d_reference(x, w, b), lut)
            for tile in range(1, 6):
                actual = self.native.conv2d_wsilu4(x.cuda(), w.cuda(), b.cuda() if b is not None else None,
                                                  lut.cuda(), tile).cpu()
                self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(ops.wsilu4(ops.conv2d_reference(*cases[0]), lut).item(), -2)

    def test_cpu_and_older_extension_fallback(self):
        x = torch.arange(-9, 9, dtype=torch.int16).reshape(1, 2, 3, 3)
        w = torch.arange(-8, 8, dtype=torch.int16).reshape(8, 2, 1, 1)
        lut = torch.arange(-32768, 32768, dtype=torch.int16)
        expected = ops.wsilu4(ops.conv2d_reference(x, w), lut)
        self.assertTrue(torch.equal(expected, ops.conv2d_wsilu4(x, w, None, lut)))
        with patch.object(self.native, 'conv2d_wsilu4', None):
            self.assertTrue(torch.equal(expected, ops.conv2d_wsilu4(x.cuda(), w.cuda(), None, lut.cuda()).cpu()))

    def test_packed_spatial_convolutions_match_independent_cpu(self):
        gen = torch.Generator().manual_seed(991)
        for shape, co, kernel, stride, padding in (
                ((2, 35, 7, 11), 129, 3, (2, 1), (1, 0)),
                ((1, 64, 9, 9), 128, 3, (2, 2), (1, 1)),
                ((1, 256, 5, 7), 129, 1, (2, 1), (1, 0))):
            for bound in (90, 32768):
                x = torch.randint(-bound, bound, shape, dtype=torch.int16, generator=gen)
                w = torch.randint(-bound, bound, (co, shape[1], kernel, kernel), dtype=torch.int16, generator=gen)
                b = torch.randint(-100, 101, (co,), dtype=torch.int16, generator=gen)
                expected = ops.conv2d_reference(x, w, b, stride, padding)
                actual = ops.conv2d(x.cuda(), w.cuda(), b.cuda(), stride, padding).cpu()
                self.assertTrue(torch.equal(expected, actual))


if __name__ == '__main__':
    unittest.main()
