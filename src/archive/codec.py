"""UF FP16 bitstream backend with bounded frame/chunk buffering."""

import hashlib
import io
from pathlib import Path
import time

import numpy as np
import torch

from src.models.image_model import DMCI
from src.utils.common import ModelStructure, get_state_dict, set_torch_env
from src.utils.stream_helper import (NalType, read_header, read_sps_remaining,
                                     read_ip_remaining, write_ip, write_sps)
from src.utils.transforms import yuv_444_to_420


class Codec:
    def __init__(self, image_path, video_path, variant='hts', device=0):
        set_torch_env()
        self.device = torch.device(f'cuda:{device}')
        torch.cuda.set_device(self.device)
        self.stream = torch.cuda.Stream(device=self.device)
        torch.cuda.set_stream(self.stream)
        self.variant = variant
        self.chunk_size = 1 if variant == 'ld' else 8
        self.image = DMCI().eval()
        self.image.load_state_dict(get_state_dict(image_path), strict=True)
        self.image.update(0)
        self.image.half().to(self.device, memory_format=torch.channels_last)
        if variant == 'ld':
            from src.models.video_model_ld import DMC
            self.video = DMC().eval()
        else:
            from src.models.video_model_ht import DMC
            self.video = DMC(model_structure=ModelStructure(variant)).eval()
        self.video.load_state_dict(get_state_dict(video_path), strict=True)
        self.video.update(0)
        self.video.half().to(self.device, memory_format=torch.channels_last)

    def tensor(self, frames):
        # Reproduce upstream's byte -> FP16 -> /255 -> -0.5 input boundary.
        array = np.concatenate(frames, axis=0)
        return (torch.from_numpy(array).unsqueeze(0).to(self.device).half() / 255.0 - 0.5
                ).contiguous(memory_format=torch.channels_last)

    def write_preamble(self, stream):
        pass

    def read_preamble(self, stream):
        if stream.peek(4)[:4] == b'UF16':
            raise ValueError('INT16 bitstream cannot be decoded with the FP16 runtime')

    @torch.no_grad()
    def encode(self, reader, path, qp_i, qp_p, reset_interval=32, intra_period=-1,
               max_frames=None, progress=lambda **values: None):
        if intra_period > 1 and intra_period % self.chunk_size:
            raise ValueError('intra_period must be 1 or a multiple of the UF chunk size')
        width, height = reader.width, reader.height
        pad_r, pad_b = DMCI.get_padding_size(height, width, 16)
        count, packets, started = 0, 0, time.monotonic()
        with open(path, 'xb') as output:
            self.write_preamble(output)
            write_sps(output, {'sps_id': 0, 'width': width, 'height': height})
            while max_frames is None or count < max_frames:
                is_i = count == 0 or intra_period == 1 or (intra_period > 1 and count != 1 and count % intra_period == 1)
                maximum = 1 if is_i else self.chunk_size
                if max_frames is not None:
                    maximum = min(maximum, max_frames - count)
                frames = []
                for _ in range(maximum):
                    frame = reader.read()
                    if frame is None:
                        break
                    frames.append(frame)
                if not frames:
                    break
                valid = len(frames)
                if not is_i:
                    frames += [frames[-1]] * (self.chunk_size - valid)
                x = self.tensor(frames)
                reset = not is_i and reset_interval > 0 and (count + self.chunk_size) % reset_interval == 1
                if is_i:
                    result = self.image.compress(x, qp_i, pad_b, pad_r)
                    self.video.clear_dpb()
                    self.video.add_ref_feature_from_frame(result['x_hat'])
                else:
                    result = self.video.compress(x, qp_p, reset, pad_b, pad_r)
                write_ip(output, is_i, 0, qp_i if is_i else qp_p, result['ec_parallel'],
                         int(reset), result['bit_stream'])
                count += valid
                packets += 1
                progress(frames=count, fps=count / max(time.monotonic() - started, 1e-9))
            output.flush()
            import os
            os.fsync(output.fileno())
        if count == 0:
            raise ValueError('Input yielded no video frames')
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        return {'frames': count, 'packets': packets, 'encode_seconds': time.monotonic() - started}

    @staticmethod
    def raw_frame(frame, width, height):
        y, uv = yuv_444_to_420(frame[:, :, :height, :width] + 0.5)
        y = (y * 255).clamp(0, 255).round().byte().squeeze(0).cpu().numpy()
        uv = (uv * 255).clamp(0, 255).byte().squeeze(0).cpu().numpy()
        return y.tobytes() + uv.tobytes()

    @torch.no_grad()
    def decode(self, path, metadata, sink=None, progress=lambda **values: None):
        count, packets, digest = 0, 0, hashlib.sha256()
        total = metadata['frames']
        started = time.monotonic()
        with open(path, 'rb') as stream:
            self.read_preamble(stream)
            sps_table = {}
            first = True
            while stream.peek(1):
                header = read_header(stream)
                if header['nal_type'] == NalType.NAL_SPS:
                    sps = read_sps_remaining(stream, header['sps_id'])
                    if (sps['width'], sps['height']) != (metadata['width'], metadata['height']):
                        raise ValueError('Bitstream dimensions differ from archive metadata')
                    sps_table[header['sps_id']] = sps
                    continue
                if count >= total:
                    raise ValueError('Unexpected extra coded packet')
                sps = sps_table[header['sps_id']]
                qp, ec_part, reset, bits = read_ip_remaining(stream)
                is_i = header['nal_type'] == NalType.NAL_I
                if first and not is_i:
                    raise ValueError('UF archive must start with an I-frame')
                first = False
                if is_i:
                    result = self.image.decompress(bits, sps, qp, ec_part)
                    self.video.clear_dpb()
                    self.video.add_ref_feature_from_frame(result['x_hat'], apply_feature_adaptor=False)
                else:
                    result = self.video.decompress(bits, sps, qp, ec_part, reset)
                frames = result['x_hat'] if isinstance(result['x_hat'], list) else [result['x_hat']]
                if len(frames) != (1 if is_i else self.chunk_size):
                    raise ValueError('UF decoder returned an unexpected chunk length')
                for frame in frames[:total-count]:
                    if not torch.isfinite(frame).all():
                        raise ValueError('Decoder produced non-finite pixels')
                    raw = self.raw_frame(frame, sps['width'], sps['height'])
                    digest.update(raw)
                    if sink is not None:
                        sink.write(raw)
                    count += 1
                packets += 1
                progress(frames=count, total=total)
        if count != total or packets != metadata['packets']:
            raise ValueError(f'Incomplete UF stream: decoded {count}/{total} frames')
        return {'decoded_frames': count, 'decoded_yuv_sha256': digest.hexdigest(),
                'decode_seconds': time.monotonic() - started}
