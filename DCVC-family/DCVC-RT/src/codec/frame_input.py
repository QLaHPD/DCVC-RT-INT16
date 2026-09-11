"""Reusable, exact INT16 input preparation for decoded 8-bit YUV420 video."""
import numpy as np
import torch

from ..layers.int16_inference import feature_to_int16


class Int16FrameInput:
    """One video/stream buffer, valid until the next prepare call.

    The lookup table reproduces the existing float32 normalization -> float16 ->
    INT16 path on the selected GPU. Chroma replication and edge padding commute
    with this per-sample conversion. No model arithmetic or color transform changes.
    """

    def __init__(self, width, height, padding_r, padding_b, device):
        self.width, self.height = width, height
        self.stream = torch.cuda.current_stream(device)
        values = (torch.arange(256, device=device, dtype=torch.float32) / 255).half()
        self.lut = feature_to_int16(values).cpu().numpy()
        self.host = torch.empty((1, 3, height + padding_b, width + padding_r),
                                dtype=torch.int16, pin_memory=True)
        self.array = self.host.numpy()[0]
        self.output = torch.empty_like(self.host, device=device)
        self.upload_done = torch.cuda.Event()
        self.pending = False

    def prepare(self, y, u, v):
        # The CPU may not overwrite pinned memory while its last DMA is pending.
        if self.pending:
            self.upload_done.synchronize()
        w, h = self.width, self.height
        np.take(self.lut, np.frombuffer(y, np.uint8).reshape(h, w),
                out=self.array[0, :h, :w])
        for plane, data in ((1, u), (2, v)):
            samples = self.lut[np.frombuffer(data, np.uint8).reshape(h // 2, w // 2)]
            for row in (0, 1):
                for col in (0, 1):
                    self.array[plane, row:h:2, col:w:2] = samples
        self.array[:, :h, w:] = self.array[:, :h, w-1:w]
        self.array[:, h:, :] = self.array[:, h-1:h, :]
        with torch.cuda.stream(self.stream):
            self.output.copy_(self.host, non_blocking=True)
            self.upload_done.record()
        self.pending = True
        return self.output

    def close(self):
        if self.pending:
            self.upload_done.synchronize()
            self.pending = False
