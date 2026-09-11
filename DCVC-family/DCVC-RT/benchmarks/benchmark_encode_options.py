#!/usr/bin/env python3
"""Paired pipeline experiments using one persistent worker and identical input."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

os.environ['DCVC_USE_INT16'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from src.cli.encode_workflow import NeuralEncoder, EncoderCfg, build_ffmpeg_chain_local
from src.codec.frame_decoder import build_bitstream_index

CASES = {
    'baseline': ('encoder', False, False),
    'input': ('encoder', True, False),
    'async': ('encoder', False, True),
    'input_async': ('encoder', True, True),
    'full': ('full', False, False),
    'full_input': ('full', True, False),
    'full_async': ('full', False, True),
    'all': ('full', True, True),
    'stages': ('stages', False, False),
    'stages_input_async': ('stages', True, True),
}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=400)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--cases', nargs='+', choices=CASES, default=['baseline', 'input'])
    args = parser.parse_args()
    if args.frames < 2 or args.rounds < 1:
        parser.error('--frames must be at least 2 and --rounds must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    probe = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=width,height', '-of', 'json', str(args.fixture)]))['streams'][0]
    width, height = probe['width'], probe['height']
    cfg = EncoderCfg(qp_i=35, qp_p=14, resolution=None, fps=24, ff_hwaccel='none')
    encoder = NeuralEncoder(torch.device('cuda:0'), str(ROOT/'checkpoints/cvpr2025_image.pth.tar'),
                            str(ROOT/'checkpoints/cvpr2025_video.pth.tar'), cfg)
    if any(CASES[case][2] for case in args.cases) and \
            not hasattr(encoder.p_net.entropy_coder, '_async_transfers'):
        parser.error('async cases require benchmarks/experiments/async_entropy.patch')
    if any(CASES[case][0] in {'full', 'stages'} for case in args.cases) and \
            not hasattr(encoder.p_net, '_tail_graph'):
        parser.error('full/stages cases require benchmarks/experiments/graph_boundaries.patch')

    def run(name, case):
        graph, fast_input, async_entropy = CASES[case]
        os.environ['DCVC_INT16_FAST_INPUT'] = str(int(fast_input))
        encoder.p_net._encode_graph_mode = graph
        for model in (encoder.i_net, encoder.p_net):
            if hasattr(model.entropy_coder, '_async_transfers'):
                model.entropy_coder._async_transfers = async_entropy
        dest = args.output/name
        dest.mkdir()
        command = build_ffmpeg_chain_local(str(args.fixture), width, height, cfg)
        command[-1:-1] = ['-frames:v', str(args.frames)]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with (dest/'ffmpeg.log').open('w') as err:
            proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=err)
            try:
                encoder.encode_from_ffmpeg_rawpipe(proc, width, height, dest/'stream.bin',
                                                  dest, 'synthetic', finalize_mode='atomic')
                assert proc.wait(timeout=10) == 0
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                proc.stdout.close()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        count = build_bitstream_index(dest/'stream.bin').frame_count
        assert count >= args.frames - 1
        data = (dest/'stream.bin').read_bytes()
        result = dict(case=case, frames=count, fps=count/elapsed,
                      cuda_peak_bytes=torch.cuda.max_memory_allocated(),
                      stream_sha256=hashlib.sha256(data).hexdigest())
        (dest/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        print(name, json.dumps(result), flush=True)
        return data, result

    expected, _ = run('warmup', 'baseline')
    results = {case: [] for case in args.cases}
    for trial in range(args.rounds):
        shift = (trial * 3) % len(args.cases)
        order = args.cases[shift:] + args.cases[:shift]
        for case in order:
            data, result = run(f'{trial+1}_{case}', case)
            assert data == expected, f'bitstream mismatch: {case}'
            results[case].append(result['fps'])
    summary = {case: dict(fps=values, median_fps=statistics.median(values))
               for case, values in results.items()}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
