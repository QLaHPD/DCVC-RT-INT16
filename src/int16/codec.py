"""Integer archive backend. Runtime and prepared identities are in the bitstream."""

import numpy as np
import torch

from src.archive.codec import Codec
from src.utils.common import set_torch_env
from . import ops
from .model import IntegerModel
from .prepared import resolve_prepared


MAGIC = b'UF16\x01\x00\x00\x00'


class IntegerCodec(Codec):
    def __init__(self, image_path, video_path, variant='hts', device=0,
                 prepared_i=None, prepared_p=None, identities=None, source_hashes=None,
                 skip_threshold=0):
        set_torch_env()
        self.device = torch.device('cpu' if device == 'cpu' else f'cuda:{device}')
        if self.device.type == 'cuda':
            import uf_int16_cuda
            if getattr(uf_int16_cuda, 'arithmetic_id', None) != ops.ARITHMETIC_ID:
                raise RuntimeError('Rebuild the UF integer CUDA extension: arithmetic version mismatch')
            torch.cuda.set_device(self.device)
            self.stream = torch.cuda.Stream(device=self.device)
            torch.cuda.set_stream(self.stream)
        else:
            self.stream = None
        loaded = []
        for index, (source, kind, path) in enumerate(((image_path, 'image', prepared_i),
                                                     (video_path, variant, prepared_p))):
            _, data = resolve_prepared(source, kind, path,
                                        identities[index] if identities else None,
                                        source_hashes[index] if source_hashes else None)
            loaded.append(data)
        self.preamble = MAGIC + b''.join(bytes.fromhex(data['identity']) for data in loaded)
        self.variant = variant
        self.chunk_size = 1 if variant == 'ld' else 8
        self.image = IntegerModel(loaded[0], self.device)
        self.video = IntegerModel(loaded[1], self.device, reconstruct_on_encode=False)
        self.set_skip_threshold(skip_threshold)

    def set_skip_threshold(self, value):
        threshold = int(float(value) * ops.FEATURE_SCALE + 0.5)
        if threshold < 0 or threshold > 32767:
            raise ValueError('skip_threshold is outside the INT16 feature range')
        self.image.skip_threshold = self.video.skip_threshold = threshold

    def tensor(self, frames):
        values = torch.from_numpy(np.concatenate(frames, axis=0)).unsqueeze(0).to(self.device)
        return ops.from_bytes(values)

    def write_preamble(self, stream):
        stream.write(self.preamble)

    def read_preamble(self, stream):
        if stream.read(len(self.preamble)) != self.preamble:
            raise ValueError('Integer bitstream runtime/prepared-model identity mismatch')

    @staticmethod
    def raw_frame(frame, width, height):
        values = frame[0, :, :height, :width].long()
        y = ops.divide_round((values[0]+ops.FEATURE_SCALE//2)*255, ops.FEATURE_SCALE)
        uv = values[1:]
        summed = uv[:, 0::2, 0::2] + uv[:, 1::2, 0::2] + uv[:, 0::2, 1::2] + uv[:, 1::2, 1::2]
        uv = torch.div((summed+2*ops.FEATURE_SCALE)*255, 4*ops.FEATURE_SCALE, rounding_mode='floor')
        return y.clamp(0,255).byte().cpu().numpy().tobytes() + uv.clamp(0,255).byte().cpu().numpy().tobytes()
