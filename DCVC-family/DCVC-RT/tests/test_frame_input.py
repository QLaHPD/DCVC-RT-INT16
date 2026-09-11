import unittest

import numpy as np
import torch

from src.codec.frame_input import Int16FrameInput
from src.layers.cuda_inference import replicate_pad
from src.layers.int16_inference import feature_to_int16
from src.utils.transforms import ycbcr420_to_444_np


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
class FrameInputTests(unittest.TestCase):
    def test_input_matches_original_conversion_padding_and_buffer_reuse(self):
        device = torch.device('cuda:0')
        with torch.no_grad(), torch.cuda.stream(torch.cuda.Stream(device=device)):
            for width, height in ((256, 2), (18, 10), (176, 96), (320, 180)):
                pad_r, pad_b = (-width) % 16, (-height) % 16
                converter = Int16FrameInput(width, height, pad_r, pad_b, device)
                saved = []
                try:
                    for offset in (0, 19, 255):
                        raw = ((np.arange(width * height * 3 // 2) + offset) % 256).astype(np.uint8)
                        size = width * height
                        y = raw[:size].reshape(1, height, width)
                        uv = raw[size:].reshape(2, height // 2, width // 2)
                        x = (torch.from_numpy(ycbcr420_to_444_np(y, uv)).to(device, torch.float32)
                             / 255).unsqueeze(0).half()
                        expected = feature_to_int16(replicate_pad(x, pad_b, pad_r))
                        actual = converter.prepare(y.tobytes(), uv[0].tobytes(), uv[1].tobytes())
                        # Keep a GPU consumer queued while the next call reuses the host buffer.
                        saved.append((actual.clone(), expected))
                    self.assertTrue(all(torch.equal(a, b) for a, b in saved))
                finally:
                    converter.close()



if __name__ == '__main__':
    unittest.main()
