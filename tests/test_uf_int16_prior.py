"""Exact entropy inputs and reconstruction at the fused partition boundary."""

import unittest

import torch

from src.int16 import ops
from src.int16.prepared import make_model


class CudaPriorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest('CUDA not available')
        try:
            import uf_int16_cuda
        except ImportError:
            raise unittest.SkipTest('UF integer CUDA extension not built')
        if not hasattr(uf_int16_cuda, 'prior_encode'):
            raise unittest.SkipTest('Rebuild UF integer CUDA extension for fused priors')
        cls.native = uf_int16_cuda

    def test_partitions_match_reference_at_skip_and_rounding_boundaries(self):
        generator = torch.Generator().manual_seed(731)
        # Include batching, odd spatial tails, LD two-part and HT four-part masks.
        for variant, shape in (('image', (1, 8, 3, 5)), ('htl', (2, 8, 4, 6)),
                               ('hts', (1, 12, 3, 7)), ('ld', (1, 8, 3, 5))):
            model = make_model(variant, meta=True)
            masks = (model.get_mask_2x(*shape, 'cpu') if variant == 'ld'
                     else model.get_mask_4x(*shape, 'cpu'))
            y = torch.randint(-32768, 32768, shape, dtype=torch.int16, generator=generator)
            # Noncontiguous channel slices match the fusion-network output.
            params = torch.randint(-32768, 32768, (shape[0], shape[1]*2, *shape[2:]),
                                   dtype=torch.int16, generator=generator)
            scales, means = params.chunk(2, dim=1)
            scales[0, 0, 0, :5] = torch.tensor([-32768, -1, 0, 103, 104])
            table = (torch.arange(65536) % 256).byte()
            for threshold in (0, 103, 32767):
                expected = torch.zeros(shape, dtype=torch.int16)
                encoded = torch.zeros(shape, dtype=torch.int16, device='cuda')
                decoded = torch.zeros_like(encoded)
                gpu_y, gpu_params, gpu_table = y.cuda(), params.cuda(), table.cuda()
                gpu_scales, gpu_means = gpu_params.chunk(2, dim=1)
                for mask in masks:
                    positions = mask.reshape(-1).nonzero().flatten()
                    selected_y = y.reshape(-1)[positions].long()
                    selected_means = means.reshape(-1)[positions].long()
                    selected_scales = scales.reshape(-1)[positions]
                    active = (selected_scales > threshold) if threshold else torch.ones_like(positions, dtype=torch.bool)
                    symbols = ops.divide_round(selected_y-selected_means, 512).clamp(-128, 127).to(torch.int8)
                    symbols[~active] = 0
                    indexes = table[selected_scales.long()+32768]
                    packed = (symbols.short() << 8) | indexes.short()
                    expected.view(-1)[positions] = ops.saturate(symbols.long()*512+selected_means)
                    gpu_positions = positions.cuda()
                    actual = self.native.prior_encode(gpu_y, gpu_scales, gpu_means, encoded,
                                                     gpu_positions, gpu_table, threshold).cpu()
                    self.assertTrue(torch.equal(actual[active].short(), packed[active]))
                    self.assertTrue(torch.all(actual[~active] == 65536))
                    self.assertTrue(torch.equal(encoded.cpu(), expected))
                    actual_indexes = self.native.prior_indexes(gpu_scales, gpu_positions,
                                                               gpu_table, threshold).cpu()
                    self.assertTrue(torch.equal(actual_indexes[active].byte(), indexes[active]))
                    self.assertTrue(torch.all(actual_indexes[~active] == -1))
                    self.native.prior_restore(symbols.cuda(), gpu_means, decoded, gpu_positions)
                    self.assertTrue(torch.equal(decoded.cpu(), expected))

    def test_signed_ties_saturation_and_zero_threshold(self):
        y = torch.tensor([-32768, -32768, -768, -256, 256, 768, 32767, 32767],
                         dtype=torch.int16).reshape(1, 1, 1, 8)
        means = torch.tensor([32767, 0, 0, 0, 0, 0, 0, -32768],
                             dtype=torch.int16).reshape_as(y)
        scales = torch.zeros_like(y)
        table = torch.full((65536,), 255, dtype=torch.uint8, device='cuda')
        positions = torch.arange(8, device='cuda')
        restored = torch.zeros_like(y, device='cuda')
        packed = self.native.prior_encode(y.cuda(), scales.cuda(), means.cuda(), restored,
                                         positions, table, 0).cpu()
        self.assertEqual((packed >> 8).tolist(), [-128, -64, -2, -1, 1, 2, 64, 127])
        self.assertEqual(restored.cpu().flatten().tolist(),
                         [-32768, -32768, -1024, -512, 512, 1024, 32767, 32256])


if __name__ == '__main__':
    unittest.main()
