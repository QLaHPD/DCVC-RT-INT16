#!/usr/bin/env python3
"""Reproducible codec checks using an ffmpeg-generated video (never the DATA tree)."""
import argparse
import cProfile
import hashlib
import io
import json
import os
from pathlib import Path
import pstats
import resource
import subprocess
import sys
import time

os.environ['DCVC_USE_INT16'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
# A separate process can run the saved native baseline without replacing the installed library.
extension_parser = argparse.ArgumentParser(add_help=False)
extension_parser.add_argument('--extension', type=Path)
extension_args, _ = extension_parser.parse_known_args()
if extension_args.extension:
    from benchmarks.benchmark_int16_conv import load_extension
    from src.layers import extension_loader
    native = load_extension(extension_args.extension)
    extension_loader.load_inference_extensions = lambda required=(): native
from src.layers.cuda_inference import replicate_pad
from src.models.image_model import DMCI
from src.models.video_model import DMC
from src.utils.common import load_model_for_inference, set_torch_env
from src.utils.transforms import ycbcr420_to_444_np
from src.utils.stream_helper import SPSHelper, write_sps, write_ip


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--extension', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--frames', type=int, default=160)
    parser.add_argument('--qp-i', type=int, default=35)
    parser.add_argument('--qp-p', type=int, default=14)
    parser.add_argument('--intra', type=int, default=0)
    parser.add_argument('--reset', type=int, default=32)
    parser.add_argument('--two-ec', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--speed-only', action='store_true')
    parser.add_argument('--start-at', type=float, default=0)
    parser.add_argument('--go-file', type=Path)
    args = parser.parse_args()
    if args.frames <= 8:
        parser.error('--frames must be greater than the eight warmup frames')
    args.output.mkdir(parents=True, exist_ok=False)
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams',
        'v:0', '-show_entries', 'stream=width,height', '-of', 'json', str(args.fixture)]))
    w, h = (probe['streams'][0][key] for key in ('width', 'height'))
    raw = subprocess.check_output(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(args.fixture),
        '-frames:v', str(args.frames), '-pix_fmt', 'yuv420p', '-f', 'rawvideo', '-'])
    frame_size = w*h*3//2
    assert len(raw) == args.frames*frame_size
    set_torch_env()
    device = torch.device('cuda:0')
    im = load_model_for_inference(DMCI(), str(ROOT/'checkpoints/cvpr2025_image.pth.tar'), device)
    vm = load_model_for_inference(DMC(), str(ROOT/'checkpoints/cvpr2025_video.pth.tar'), device)
    im.set_use_two_entropy_coders(args.two_ec)
    vm.set_use_two_entropy_coders(args.two_ec)
    frames = []
    for i in range(args.frames):
        f = np.frombuffer(raw, np.uint8, frame_size, i*frame_size)
        x = ycbcr420_to_444_np(f[:w*h].reshape(1,h,w), f[w*h:].reshape(2,h//2,w//2))
        x = (torch.from_numpy(x).to(device, torch.float32)/255).unsqueeze(0).half()
        frames.append(replicate_pad(x, (-h)%16, (-w)%16).contiguous())
    profiler = cProfile.Profile()
    streams, metadata, features = [], [], []
    last_qp = 0
    vm.set_curr_poc(0)
    torch.cuda.synchronize()
    (args.output/'ready').touch()
    if args.go_file:
        while not args.go_file.exists():
            time.sleep(.02)
    while time.time() < args.start_at:
        time.sleep(min(.1, args.start_at-time.time()))
    encode_start_unix = time.time()
    start = time.perf_counter()
    steady_start = start
    with torch.no_grad():
        for i, x in enumerate(frames):
            if i == 8:
                torch.cuda.synchronize()
                steady_start = time.perf_counter()
            is_i = i == 0 or (args.intra > 0 and i % args.intra == 0)
            ada = int(not is_i and args.reset > 0 and i % args.reset == 1)
            if args.profile and i == 10:
                profiler.enable()
            if is_i:
                qp = args.qp_i
                enc = im.compress(x, qp)
                vm.clear_dpb()
                vm.add_ref_frame(None, enc['x_hat'])
            else:
                if ada:
                    vm.prepare_feature_adaptor_i(last_qp)
                qp = vm.shift_qp(args.qp_p, [0,1,0,2,0,2,0,2][i%8])
                enc = vm.compress(x, qp)
                last_qp = qp
            if args.profile and i == 10:
                torch.cuda.synchronize()
                profiler.disable()
            streams.append(enc['bit_stream'])
            metadata.append((is_i, qp, dict(height=h, width=w, ec_part=int(args.two_ec), use_ada_i=ada)))
            if not args.speed_only:
                ref = vm.dpb[0].feature if vm.dpb[0].feature is not None else vm.dpb[0].frame
                features.append(hashlib.sha256(ref.cpu().numpy().tobytes()).hexdigest())
        torch.cuda.synchronize()
        end = time.perf_counter()
        result = dict(frames=args.frames, width=w, height=h, qp_i=args.qp_i, qp_p=args.qp_p,
            encode_fps=args.frames/(end-start), steady_encode_fps=(args.frames-8)/(end-steady_start),
            encode_seconds=end-start, feature_hashes=features)
        result.update(encode_start_unix=encode_start_unix, encode_end_unix=time.time())
        buffer, helper = io.BytesIO(), SPSHelper()
        for stream, (is_i, qp, sps) in zip(streams, metadata):
            sps_id, fresh = helper.get_sps_id(sps)
            sps['sps_id'] = sps_id
            if fresh:
                write_sps(buffer, sps)
            write_ip(buffer, is_i, sps_id, qp, stream)
        encoded_bytes = buffer.getvalue()
        (args.output/'stream.bin').write_bytes(encoded_bytes)
        result['stream_sha256'] = hashlib.sha256(encoded_bytes).hexdigest()
        if not args.speed_only:
            vm.clear_dpb()
            vm.set_curr_poc(0)
            pixels = hashlib.sha256()
            start = time.perf_counter()
            frame_hashes = []
            for stream, (is_i, qp, sps) in zip(streams, metadata):
                if is_i:
                    dec = im.decompress(stream, sps, qp)
                    vm.clear_dpb()
                    vm.add_ref_frame(None, dec['x_hat'])
                else:
                    if sps['use_ada_i']:
                        vm.reset_ref_feature()
                    dec = vm.decompress(stream, sps, qp)
                data = dec['x_hat'].cpu().numpy().tobytes()
                pixels.update(data)
                frame_hashes.append(hashlib.sha256(data).hexdigest())
            torch.cuda.synchronize()
            result.update(decode_fps=args.frames/(time.perf_counter()-start),
                          pixels_sha256=pixels.hexdigest(), frame_hashes=frame_hashes)
    result.update(max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  cuda_peak_bytes=torch.cuda.max_memory_allocated())
    if args.compare:
        expected = json.loads((args.compare/'result.json').read_text())
        for key in ('stream_sha256', 'feature_hashes', 'frame_hashes', 'pixels_sha256'):
            if key in result:
                assert result[key] == expected[key], f'bit-exact mismatch: {key}'
        assert encoded_bytes == (args.compare/'stream.bin').read_bytes()
        result['exact'] = True
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    if args.profile:
        profiler.dump_stats(str(args.output/'encode.prof'))
        with (args.output/'profile.txt').open('w') as f:
            pstats.Stats(profiler, stream=f).sort_stats('cumtime').print_stats(45)
    print(json.dumps({k:v for k,v in result.items() if not k.endswith('_hashes')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
