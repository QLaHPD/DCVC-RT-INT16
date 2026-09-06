#!/usr/bin/env python3
"""Measure the production FFmpeg/prefetch/encode/file-writing path on a synthetic fixture.

Uses NeuralEncoder directly so channel cleanup is never invoked by this benchmark.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
os.environ['DCVC_USE_INT16'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
p = argparse.ArgumentParser(__doc__)
p.add_argument('--extension',type=Path,required=True)
p.add_argument('--fixture',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--frames',type=int,default=160)
p.add_argument('--go-file',type=Path)
p.add_argument('--speed-only',action='store_true',help=argparse.SUPPRESS)
a = p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
import torch
from benchmarks.benchmark_int16_conv import load_extension
from src.layers import extension_loader
native = load_extension(a.extension)
extension_loader.load_inference_extensions = lambda required=(): native
from src.cli.encode_workflow import NeuralEncoder, EncoderCfg, build_ffmpeg_chain_local
from src.codec.frame_decoder import build_bitstream_index
probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
    '-show_entries','stream=width,height','-of','json',str(a.fixture)]))
w,h = (probe['streams'][0][key] for key in ('width','height'))
cfg = EncoderCfg(qp_i=35,qp_p=14,resolution=None,fps=24,ff_hwaccel='none')
encoder = NeuralEncoder(torch.device('cuda:0'),str(ROOT/'checkpoints/cvpr2025_image.pth.tar'),
    str(ROOT/'checkpoints/cvpr2025_video.pth.tar'),cfg)
cmd = build_ffmpeg_chain_local(str(a.fixture),w,h,cfg)
cmd[-1:-1] = ['-frames:v',str(a.frames)]
torch.cuda.synchronize()
(a.output/'ready').touch()
if a.go_file:
    while not a.go_file.exists():
        time.sleep(.02)
start = time.time()
with (a.output/'ffmpeg.log').open('w') as err:
    proc = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=err)
    try:
        encoder.encode_from_ffmpeg_rawpipe(proc,w,h,a.output/'stream.bin',a.output,'synthetic',finalize_mode='atomic')
        assert proc.wait(timeout=10) == 0
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        proc.stdout.close()
torch.cuda.synchronize()
end = time.time()
count = build_bitstream_index(a.output/'stream.bin').frame_count
assert count >= a.frames-1, f'Unexpected frame count: {count}'
data = dict(frames=count,encode_fps=count/(end-start),steady_encode_fps=count/(end-start),
    encode_start_unix=start,encode_end_unix=end,
    max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    cuda_peak_bytes=torch.cuda.max_memory_allocated(),
    stream_sha256=hashlib.sha256((a.output/'stream.bin').read_bytes()).hexdigest(),
    timing='Full FFmpeg/prefetch/conversion/encode/atomic-write path, includes first frame')
(a.output/'result.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(data,indent=2),flush=True)
