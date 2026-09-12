import unittest
import torch
from src.int16.model import IntegerModel
from src.int16.prepared import make_model


class PriorPositionTests(unittest.TestCase):
    def test_positions_preserve_partition_order_and_replace_changed_shapes(self):
        for variant in ('image', 'hts', 'htl', 'ld'):
            model = IntegerModel.__new__(IntegerModel)
            model.variant = variant
            model.device = torch.device('cpu')
            model.model = make_model(variant, meta=True)
            model._position_shape = model._positions = None
            previous = None
            for shape in ((1, 8, 3, 5), (2, 8, 5, 3), (1, 8, 3, 5)):
                values = torch.arange(torch.tensor(shape).prod()).reshape(shape)
                masks = (model.model.get_mask_2x(*shape, model.device) if variant == 'ld'
                         else model.model.get_mask_4x(*shape, model.device))
                positions = model._mask_positions(shape)
                self.assertIsNot(positions, previous)
                self.assertIs(positions, model._mask_positions(shape))
                for mask, position in zip(masks, positions):
                    self.assertTrue(torch.equal(values[mask], values.flatten().index_select(0, position)))
                self.assertEqual(torch.cat(positions).sort().values.tolist(), list(range(values.numel())))
                previous = positions
