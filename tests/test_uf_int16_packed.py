"""Packed INT16 kernels retain the original arithmetic and pixel boundary."""

import unittest
from unittest.mock import patch

import torch

from src.int16 import ops
from src.int16.codec import IntegerCodec


class PackedIntegerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest('CUDA not available')
        try:
            import uf_int16_cuda
        except ImportError:
            raise unittest.SkipTest('UF integer extension not built')
        if not all(hasattr(uf_int16_cuda, name) for name in ('conv2d_post', 'raw_frame')):
            raise unittest.SkipTest('Rebuild UF integer extension for packed kernels')
        cls.native = uf_int16_cuda
        cls.direct_tiles = (7, 8)
        if callable(getattr(uf_int16_cuda, 'depth_conv_block', None)):
            cls.direct_tiles += (9, 10, 11)

    def test_direct_tiles_match_cpu_with_batching_and_tails(self):
        gen = torch.Generator().manual_seed(1021)
        lut = torch.randint(-32768, 32768, (65536,), dtype=torch.int16, generator=gen)
        for shape, co in (((2, 32, 3, 7), 20), ((1, 96, 9, 16), 68),
                          ((1, 256, 1, 1), 4)):
            for bound in (90, 32768):
                x = torch.randint(-bound, bound, shape, dtype=torch.int16, generator=gen)
                w = torch.randint(-bound, bound, (co, shape[1], 1, 1), dtype=torch.int16, generator=gen)
                b = torch.arange(co, dtype=torch.int16)
                expected = ops.conv2d_reference(x, w, b)
                activated = ops.wsilu4(expected, lut)
                for tile in (6, *self.direct_tiles):
                    with self.subTest(shape=shape, bound=bound, tile=tile):
                        self.assertTrue(torch.equal(expected,
                            self.native.conv2d_tiled(x.cuda(), w.cuda(), b.cuda(), tile).cpu()))
                        self.assertTrue(torch.equal(activated,
                            self.native.conv2d_wsilu4(x.cuda(), w.cuda(), b.cuda(), lut.cuda(), tile).cpu()))

    def test_direct_wide_accumulation_and_cancellation(self):
        lut = torch.arange(-32768, 32768, dtype=torch.int16)
        for value in (-1, 32767, -32768):
            x = torch.full((1, 32768, 1, 1), value, dtype=torch.int16)
            w = torch.full((4, 32768, 1, 1), value, dtype=torch.int16)
            w[1].fill_(-32768)
            w[2, :16384].fill_(-value if value != -32768 else 32767)
            w[3].zero_()
            expected = ops.conv2d_reference(x, w)
            for tile in self.direct_tiles:
                self.assertTrue(torch.equal(expected,
                    self.native.conv2d_tiled(x.cuda(), w.cuda(), None, tile).cpu()))
                self.assertTrue(torch.equal(ops.wsilu4(expected, lut),
                    self.native.conv2d_wsilu4(x.cuda(), w.cuda(), None, lut.cuda(), tile).cpu()))

    def test_direct_tiles_accept_two_byte_storage_offsets(self):
        gen = torch.Generator().manual_seed(1025)
        lut = torch.arange(-32768, 32768, dtype=torch.int16)
        for shape, co in (((1, 32, 1, 1), 20), ((2, 256, 9, 16), 512)):
            x = torch.randint(-90, 91, shape, dtype=torch.int16, generator=gen)
            w = torch.randint(-90, 91, (co, shape[1], 1, 1), dtype=torch.int16, generator=gen)
            gx = torch.empty(x.numel()+1, dtype=torch.int16, device='cuda')[1:].view(x.shape)
            gw = torch.empty(w.numel()+1, dtype=torch.int16, device='cuda')[1:].view(w.shape)
            gx.copy_(x); gw.copy_(w)
            self.assertEqual(gx.data_ptr()%4, 2)
            self.assertEqual(gw.data_ptr()%4, 2)
            expected = ops.conv2d_reference(x, w)
            for tile in self.direct_tiles:
                self.assertTrue(torch.equal(expected, self.native.conv2d_tiled(gx, gw, None, tile).cpu()))
                self.assertTrue(torch.equal(ops.wsilu4(expected, lut),
                    self.native.conv2d_wsilu4(gx, gw, None, lut.cuda(), tile).cpu()))
            self.assertTrue(torch.equal(expected, ops.conv2d(gx, gw).cpu()))
            self.assertTrue(torch.equal(expected, ops.conv2d_post(gx, gw, None).cpu()))

    def test_post_operations_preserve_each_saturation(self):
        gen = torch.Generator().manual_seed(1022)
        for ci, co, height, width in ((35, 19, 3, 7), (256, 512, 9, 16)):
            x = torch.randint(-90, 91, (2, ci, height, width), dtype=torch.int16, generator=gen)
            w = torch.randint(-90, 91, (co, ci, 1, 1), dtype=torch.int16, generator=gen)
            b = torch.arange(co, dtype=torch.int16)
            lut = torch.randint(-32768, 32768, (65536,), dtype=torch.int16, generator=gen)
            # Residual inputs may be noncontiguous views, like the unfused add.
            residual_base = torch.randint(-32768, 32768, (2, co, height, width*2),
                                          dtype=torch.int16, generator=gen)
            residual = residual_base[..., ::2]
            residual_cuda = residual_base.cuda()[..., ::2]
            self.assertFalse(residual_cuda.is_contiguous())
            shortcut = -residual.clamp_min(-32767)
            conv = ops.conv2d_reference(x, w, b)
            expected = ops.add(ops.add(ops.lookup(conv, lut), residual), shortcut)
            actual = ops.conv2d_post(x.cuda(), w.cuda(), b.cuda(), lut.cuda(),
                                    residual_cuda, shortcut.cuda())
            self.assertTrue(torch.equal(expected, actual.cpu()))
            with patch.object(self.native, 'conv2d_post', None):
                self.assertTrue(torch.equal(expected,
                    ops.conv2d_post(x.cuda(), w.cuda(), b.cuda(), lut.cuda(),
                                    residual_cuda, shortcut.cuda()).cpu()))
        x = torch.zeros((1, 1, 1, 1), dtype=torch.int16)
        w = x.clone()
        b = torch.tensor([32767], dtype=torch.int16)
        residual = torch.full_like(x, 32767)
        shortcut = torch.full_like(x, -32768)
        self.assertEqual(ops.conv2d_post(x, w, b, residual=residual, shortcut=shortcut).item(), -1)
        self.assertEqual(ops.conv2d_post(x.cuda(), w.cuda(), b.cuda(),
                         residual=residual.cuda(), shortcut=shortcut.cuda()).item(), -1)

    def test_specialized_depthwise_matches_cpu(self):
        gen = torch.Generator().manual_seed(1023)
        for shape in ((2, 5, 1, 1), (2, 5, 7, 11), (1, 32, 3, 7)):
            for bound in (90, 32768):
                x = torch.randint(-bound, bound, shape, dtype=torch.int16, generator=gen)
                w = torch.randint(-bound, bound, (shape[1], 1, 3, 3), dtype=torch.int16, generator=gen)
                b = torch.arange(shape[1], dtype=torch.int16)
                expected = ops.conv2d_reference(x, w, b, padding=(1, 1), groups=shape[1])
                actual = ops.conv2d(x.cuda(), w.cuda(), b.cuda(), padding=(1, 1), groups=shape[1])
                self.assertTrue(torch.equal(expected, actual.cpu()))

    def test_raw_pixel_conversion_matches_cpu_with_crops(self):
        gen = torch.Generator().manual_seed(1024)
        cases = [(torch.arange(-32768, 32768, dtype=torch.int16).repeat(3).reshape(1, 3, 256, 256), 256, 256)]
        for bound in (512, 32768):
            x = torch.randint(-bound, bound, (1, 3, 10, 14), dtype=torch.int16, generator=gen)
            cases.extend(((x, 10, 8), (x[..., ::2], 6, 8),
                          (x.transpose(-1, -2), 8, 12), (x, 2, 2)))
        for x, width, height in cases:
            with self.subTest(shape=x.shape, width=width, height=height):
                expected = IntegerCodec.raw_frame(x, width, height)
                self.assertEqual(expected, IntegerCodec.raw_frame(x.cuda(), width, height))

    def test_pixel_conversion_older_extension_fallback(self):
        x = torch.arange(-90, 90, dtype=torch.int16).reshape(1, 3, 6, 10)
        expected = IntegerCodec.raw_frame(x, 8, 4)
        with patch.object(self.native, 'raw_frame', None):
            self.assertEqual(expected, IntegerCodec.raw_frame(x.cuda(), 8, 4))

    def test_native_depth_block_matches_independent_cpu_stages(self):
        if not callable(getattr(self.native, 'depth_conv_block', None)):
            self.skipTest('Rebuild for native depth blocks')
        gen = torch.Generator().manual_seed(1042)
        for ratio in (1, 2):
            channels = 20
            sizes = ((channels//ratio, channels, 1, 1), (channels//ratio, 1, 3, 3),
                     (channels, channels//ratio, 1, 1), (channels*4//ratio, channels, 1, 1),
                     (channels, channels//ratio, 1, 1))
            for bound in (90, 32768):
                x = torch.randint(-bound, bound, (2, channels, 3, 7), dtype=torch.int16, generator=gen)
                weights = [torch.randint(-bound, bound, shape, dtype=torch.int16, generator=gen)
                           for shape in sizes]
                biases = [None if i%2 else torch.randint(-bound, bound, (shape[0],),
                          dtype=torch.int16, generator=gen) for i, shape in enumerate(sizes)]
                lut = torch.randint(-32768, 32768, (65536,), dtype=torch.int16, generator=gen)
                first = ops.lookup(ops.conv2d_reference(x, weights[0], biases[0]), lut)
                spatial = ops.conv2d_reference(first, weights[1], biases[1],
                                               padding=(1, 1), groups=channels//ratio)
                residual = ops.add(ops.conv2d_reference(spatial, weights[2], biases[2]), x)
                expanded = ops.wsilu4(ops.conv2d_reference(residual, weights[3], biases[3]), lut)
                expected = ops.add(ops.conv2d_reference(expanded, weights[4], biases[4]), residual)
                for shortcut in (False, True):
                    actual = self.native.depth_conv_block(x.cuda(), [w.cuda() for w in weights],
                        [b.cuda() if b is not None else None for b in biases], lut.cuda(), shortcut)
                    self.assertTrue(torch.equal(ops.add(expected, x) if shortcut else expected, actual.cpu()))

    def test_layer_native_depth_block_selection_and_fallback(self):
        from src.int16.layers import Layers
        from src.layers.layers import DepthConvBlock
        if not callable(getattr(self.native, 'depth_conv_block', None)):
            self.skipTest('Rebuild for native depth blocks')
        gen = torch.Generator().manual_seed(1043)
        for ratio in (1, 2):
            block = DepthConvBlock(12, 20, dcb2=ratio==2, shortcut=ratio==1)
            tensors = {name: torch.randint(-90, 91, value.shape, dtype=torch.int16, generator=gen)
                       for name, value in block.named_parameters()}
            tensors['tables.wsilu'] = torch.randint(-32768, 32768, (65536,),
                                                     dtype=torch.int16, generator=gen)
            prepared = dict(tensors=tensors)
            x = torch.randint(-90, 91, (2, 12, 3, 7), dtype=torch.int16, generator=gen)
            reference = Layers(block, prepared, 'cpu')(block, x)
            current = Layers(block, prepared, 'cuda')
            self.assertIn(id(block), current._native_blocks)
            self.assertTrue(torch.equal(reference, current(block, x.cuda()).cpu()))
            with patch.object(self.native, 'depth_conv_block', None):
                fallback = Layers(block, prepared, 'cuda')
                self.assertFalse(fallback._native_blocks)
                self.assertTrue(torch.equal(reference, fallback(block, x.cuda()).cpu()))


if __name__ == '__main__':
    unittest.main()
