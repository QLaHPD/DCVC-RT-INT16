"""Integer archive backend. Runtime and prepared identities are in the bitstream."""

import numpy as np
import torch

from src.archive.codec import Codec
from src.utils.common import set_torch_env
from . import ops
from .model import IntegerModel
from .prepared import PreparedFileGuard, resolve_prepared


MAGIC = b'UF16\x01\x00\x00\x00'


class IntegerCodec(Codec):
    def __init__(self, image_path, video_path, variant='hts', device=0,
                 prepared_i=None, prepared_p=None, identities=None, source_hashes=None,
                 skip_threshold=0, image_only=False):
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
        loaded_identities = []
        self.image = self.video = None
        specifications = [(image_path, 'image', prepared_i)]
        if not image_only:
            specifications.append((video_path, variant, prepared_p))
        for index, (source, kind, path) in enumerate(specifications):
            guard = PreparedFileGuard()
            options = ({'memory_map': True, '_guard': guard}
                       if self.device.type == 'cuda' else {})
            _, data = resolve_prepared(source, kind, path,
                                        identities[index] if identities else None,
                                        source_hashes[index] if source_hashes else None, **options)
            loaded_identities.append(bytes.fromhex(data['identity']))
            if index == 0:
                self.image = IntegerModel(data, self.device)
            else:
                self.video = IntegerModel(data, self.device, reconstruct_on_encode=False)
            guard.check()
            # CUDA layers own their device copies. Release this CPU checkpoint
            # before reading the next one instead of retaining both throughout.
            del data
        self.preamble = MAGIC + b''.join(loaded_identities)
        if image_only:
            self.preamble += bytes(32)
        self.variant = variant
        self.chunk_size = 1 if variant == 'ld' else 8
        self.set_skip_threshold(skip_threshold)

    def set_skip_threshold(self, value):
        threshold = int(float(value) * ops.FEATURE_SCALE + 0.5)
        if threshold < 0 or threshold > 32767:
            raise ValueError('skip_threshold is outside the INT16 feature range')
        self.image.skip_threshold = threshold
        if self.video is not None:
            self.video.skip_threshold = threshold

    def set_gpu_gate(self, path):
        from .scheduling import GpuGate
        gate = GpuGate(path) if path and self.device.type == 'cuda' and self.variant == 'htl' else None
        self.image.gpu_gate = gate
        if self.video is not None:
            self.video.gpu_gate = gate

    @staticmethod
    def _expand_yuv420_frame(frame):
        if frame.ndim != 2:
            return frame
        h = frame.shape[0] * 2 // 3
        w = frame.shape[1]
        flat = frame.reshape(-1)
        n = h * w
        y = flat[:n].reshape(h, w)
        u = flat[n:n + n // 4].reshape(h // 2, w // 2)
        v = flat[n + n // 4:].reshape(h // 2, w // 2)
        return np.stack((y, u.repeat(2, 0).repeat(2, 1), v.repeat(2, 0).repeat(2, 1)))

    def _supports_yuv420_encode(self):
        if self.device.type != 'cuda':
            return False
        if not all(getattr(type(model), 'supports_byte_input', False) and
                   getattr(type(model), 'supports_yuv420_input', False) and
                   getattr(type(model.layers), 'supports_byte_input', False)
                   for model in (self.image, self.video) if model is not None):
            return False
        import uf_int16_cuda
        return (callable(getattr(uf_int16_cuda, 'from_bytes_unshuffle', None)) and
                bool(getattr(uf_int16_cuda, 'supports_yuv420_unshuffle', False)))

    def encode(self, reader, *args, **kwargs):
        if hasattr(reader, 'raw_yuv420') and self._supports_yuv420_encode():
            previous = reader.raw_yuv420
            reader.raw_yuv420 = True
            try:
                return super().encode(reader, *args, **kwargs)
            finally:
                reader.raw_yuv420 = previous
        return super().encode(reader, *args, **kwargs)

    def tensor(self, frames):
        if frames and frames[0].ndim == 2:
            frames = [self._expand_yuv420_frame(f) for f in frames]
        values = torch.from_numpy(np.concatenate(frames, axis=0)).unsqueeze(0).to(self.device)
        return ops.from_bytes(values)

    def tensor_for_encode(self, frames):
        if frames and frames[0].ndim == 2:
            if self._supports_yuv420_encode():
                rows, w = frames[0].shape
                h_half = rows // 3
                t = len(frames)
                if t == 1:
                    return torch.from_numpy(frames[0]).view(1, 1, 3, h_half, w).to(self.device)
                staging = getattr(self, '_staging_yuv420', None)
                if staging is None or staging.shape != (t, rows, w):
                    staging = np.empty((t, rows, w), dtype=np.uint8)
                    self._staging_yuv420 = staging
                np.stack(frames, axis=0, out=staging)
                return torch.from_numpy(staging).view(1, t, 3, h_half, w).to(self.device)
            frames = [self._expand_yuv420_frame(f) for f in frames]
        if self.device.type == 'cuda' and all(getattr(type(model), 'supports_byte_input', False) and
                getattr(type(model.layers), 'supports_byte_input', False)
                for model in (self.image, self.video) if model is not None):
            import uf_int16_cuda
            if callable(getattr(uf_int16_cuda, 'from_bytes_unshuffle', None)):
                return torch.from_numpy(np.concatenate(frames, axis=0)).unsqueeze(0).to(self.device)
        return self.tensor(frames)

    def write_preamble(self, stream):
        stream.write(self.preamble)

    def read_preamble(self, stream):
        if stream.read(len(self.preamble)) != self.preamble:
            raise ValueError('Integer bitstream runtime/prepared-model identity mismatch')

    def decode_video(self, bits, sps, qp, ec_part, reset, remaining):
        if (0 < remaining < self.chunk_size
                and getattr(type(self.video), 'supports_partial_output', False)
                and getattr(type(self.video.layers), 'supports_partial_output', False)):
            return self.video.decompress(bits, sps, qp, ec_part, reset,
                                         output_frames=remaining), remaining
        return super().decode_video(bits, sps, qp, ec_part, reset, remaining)

    def skip_video(self, bits, sps, qp, ec_part, reset):
        if not getattr(type(self.video), 'supports_skipped_output', False):
            return False
        result = self.video.decompress(bits, sps, qp, ec_part, reset, output_frames=0)
        if not isinstance(result['x_hat'], list) or result['x_hat']:
            raise ValueError('UF decoder returned pixels for a skipped chunk')
        return True

    @staticmethod
    def raw_frame(frame, width, height):
        if frame.is_cuda:
            import uf_int16_cuda
            fused = getattr(uf_int16_cuda, 'raw_frame', None)
            if fused is not None:
                return fused(frame, width, height).cpu().numpy().tobytes()
        values = frame[0, :, :height, :width].long()
        y = ops.divide_round((values[0]+ops.FEATURE_SCALE//2)*255, ops.FEATURE_SCALE)
        uv = values[1:]
        summed = uv[:, 0::2, 0::2] + uv[:, 1::2, 0::2] + uv[:, 0::2, 1::2] + uv[:, 1::2, 1::2]
        uv = torch.div((summed+2*ops.FEATURE_SCALE)*255, 4*ops.FEATURE_SCALE, rounding_mode='floor')
        return y.clamp(0,255).byte().cpu().numpy().tobytes() + uv.clamp(0,255).byte().cpu().numpy().tobytes()

    @classmethod
    def raw_frames(cls, frames, width, height):
        if frames and len(frames) <= 8 and all(frame.is_cuda for frame in frames):
            import uf_int16_cuda
            fused = getattr(uf_int16_cuda, 'raw_frames', None)
            if fused is not None:
                return fused(frames, width, height).cpu().numpy().tobytes()
        return super().raw_frames(frames, width, height)
