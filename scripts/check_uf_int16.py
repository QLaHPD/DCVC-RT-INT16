#!/usr/bin/env python3
"""Compare UF integer bitstreams/state across runs, CPU reference and physical GPUs."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from src.int16.model import IntegerModel
from src.int16.ops import ARITHMETIC_ID
from src.int16.prepared import resolve_prepared


class IntegerOnly(TorchDispatchMode):
    """Reject floating tensors in inference, including accidental float fallback."""
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        def check(values):
            for value in tree_flatten(values)[0]:
                if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
                    raise AssertionError(f'Floating-point tensor in integer inference: {func}')
        check((args, kwargs))
        result = func(*args, **(kwargs or {}))
        check(result)
        return result


def tensor_hash(value):
    if value is None:
        return None
    if isinstance(value, list):
        return [tensor_hash(item) for item in value]
    raw = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(raw.astype(raw.dtype.newbyteorder('<'), copy=False).tobytes()).hexdigest()


def state_record(model, frames, bits):
    return {'bitstream': hashlib.sha256(bits).hexdigest(), 'bytes': len(bits),
            'reconstruction': tensor_hash(frames),
            'state': {name: tensor_hash(getattr(model, name)) for name in ('reference', 'memory', 'context')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('hts', 'htl', 'ld'), default='hts')
    parser.add_argument('--cuda_idx', type=int, default=0)
    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--cpu-reference', action='store_true')
    parser.add_argument('--prepared_i')
    parser.add_argument('--prepared_p')
    parser.add_argument('--compare', help='Previously saved report from another run/device')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.size < 16 or args.size % 16:
        parser.error('--size must be a positive multiple of 16')
    paths, data = [], []
    for name, variant, output in (('image', 'image', args.prepared_i),
                                  ('video_'+args.variant, args.variant, args.prepared_p)):
        path, prepared = resolve_prepared(ROOT/'checkpoints'/f'cvpr2026_{name}.pth.tar', variant, output)
        paths.append(str(path))
        data.append(prepared)
    device = f'cuda:{args.cuda_idx}'
    sps = {'height': args.size, 'width': args.size}
    records, packets = [], []
    generator = torch.Generator(device='cpu').manual_seed(437)
    def source(channels):
        return torch.randint(-256, 257, (1, channels, args.size, args.size),
                             dtype=torch.int16, generator=generator).to(device)
    started = time.monotonic()
    image, video = IntegerModel(data[0], device), IntegerModel(data[1], device)
    with IntegerOnly():
        result = image.compress(source(3), 36, 0, 0)
        records.append(state_record(image, result['x_hat'], result['bit_stream']))
        packets.append((True, False, result['bit_stream']))
        video.add_ref_feature_from_frame(result['x_hat'])
        for reset in (False, True, False):
            result = video.compress(source(3 if args.variant=='ld' else 24), 30, reset, 0, 0)
            records.append(state_record(video, result['x_hat'], result['bit_stream']))
            packets.append((False, reset, result['bit_stream']))
    del image, video, result
    torch.cuda.empty_cache()
    for backend in ([device, 'cpu'] if args.cpu_reference else [device]):
        image, video = IntegerModel(data[0], backend), IntegerModel(data[1], backend)
        with IntegerOnly():
            for index, (is_i, reset, bits) in enumerate(packets):
                model = image if is_i else video
                result = model.decompress(bits, sps, 36 if is_i else 30, 1, reset)
                if state_record(model, result['x_hat'], bits) != records[index]:
                    raise AssertionError(f'{backend} diverged at packet {index}')
                if is_i:
                    video.add_ref_feature_from_frame(result['x_hat'])
                print(f'{backend}: packet {index}, reset={reset}, integer state matches', flush=True)
        del image, video, result
        torch.cuda.empty_cache()
    report = {'arithmetic': ARITHMETIC_ID, 'variant': args.variant, 'size': args.size, 'seed': 437,
              'prepared': [value['identity'] for value in data], 'records': records,
              'cpu_reference_checked': args.cpu_reference, 'device': torch.cuda.get_device_name(args.cuda_idx),
              'seconds': time.monotonic()-started}
    if args.compare:
        previous = json.loads(Path(args.compare).read_text())
        for key in ('arithmetic', 'variant', 'size', 'seed', 'prepared', 'records'):
            if previous[key] != report[key]:
                raise AssertionError(f'Result differs from comparison report: {key}')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(f'Passed. Report: {output}', flush=True)


if __name__ == '__main__':
    main()
