"""Regression for scalar INT16 rounding during CUDA Graph capture."""
import unittest
import torch
from src.layers.int16_inference import round_divide


class RoundCaptureTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_all_int16_values_capture_and_replay(self):
        x = torch.arange(-32768, 32768, device='cuda', dtype=torch.int32)
        expected = round_divide(x, torch.tensor(256, device='cuda'))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                round_divide(x, 256)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = round_divide(x, 256)
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(result, expected))
        x.neg_()
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(result, round_divide(x, 256)))
