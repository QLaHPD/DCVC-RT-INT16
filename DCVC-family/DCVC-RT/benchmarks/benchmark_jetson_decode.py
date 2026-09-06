#!/usr/bin/env python3
"""Compare software/NVDEC bitstreams on a generated fixture, one process.

Does not run channel discovery, source cleanup, or production workloads.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import sys
import time

os.environ['DCVC_USE_INT16'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--fixture', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
import torch
from src.cli.encode_workflow import EncoderCfg, EncodeTask, NeuralEncoder, process_one_file
from src.codec.frame_decoder import build_bitstream_index
cfg = EncoderCfg(qp_i=35, qp_p=14, resolution=96, fps=24, ff_hwaccel='none')
encoder = NeuralEncoder(torch.device('cuda:0'), str(ROOT/'checkpoints/cvpr2025_image.pth.tar'),
                        str(ROOT/'checkpoints/cvpr2025_video.pth.tar'), cfg)
results = []
for i, mode in enumerate(('none', 'jetson', 'jetson', 'none')):
    cfg.ff_hwaccel = mode
    output = a.output/f'{i}_{mode}'
    start = time.monotonic()
    ok, elapsed = process_one_file(EncodeTask('synthetic', str(a.fixture), True, False),
        queue.Queue(), 0, output, encoder, cfg, False, {}, 'atomic')
    assert ok, f'{mode} failed; inspect {output}'
    bits = next(output.glob('*.bin'))
    count = build_bitstream_index(bits).frame_count
    result = dict(backend=mode, frames=count, seconds=elapsed, fps=count/elapsed,
                  sha256=hashlib.sha256(bits.read_bytes()).hexdigest())
    if mode == 'jetson':
        logs = list(output.glob('*.jsonl'))
        assert logs and not any('decoder_fallback' in log.read_text() for log in logs), 'NVDEC fell back'
    results.append(result)
    print(json.dumps(result), flush=True)
assert len({r['sha256'] for r in results}) == 1, 'bitstream mismatch'
(a.output/'result.json').write_text(json.dumps(results, indent=2)+'\n')
