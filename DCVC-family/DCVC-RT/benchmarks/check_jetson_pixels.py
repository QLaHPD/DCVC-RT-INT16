#!/usr/bin/env python3
"""Compare NVDEC and software pixels after the production scale/FPS filters.

Pass generated H.264 fixtures. No model loading or source cleanup is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cli.jetson_decode import start_jetson_pipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fixtures', type=Path, nargs='+')
    args = parser.parse_args()
    for fixture in args.fixtures:
        cmd = ['ffmpeg', '-hide_banner', '-nostdin', '-loglevel', 'error', '-nostats',
               '-fflags', '+genpts', '-reinit_filter', '0', '-hwaccel', 'none', '-i', str(fixture),
               '-vf', 'scale=176:96:flags=fast_bilinear+full_chroma_int:in_color_matrix=bt709:out_color_matrix=bt709,format=yuv420p,setsar=1/1,fps=24:round=down',
               '-pix_fmt', 'yuv420p', '-vsync', '0', '-threads', '0', '-f', 'rawvideo', '-']
        start = time.monotonic()
        software = subprocess.check_output(cmd)
        sw_seconds = time.monotonic() - start
        start = time.monotonic()
        proc = start_jetson_pipe(str(fixture), cmd)
        try:
            hardware = proc.stdout.read()
            proc.wait(timeout=30)
        finally:
            proc.close()
        result = dict(fixture=fixture.name, identical=hardware == software,
                      software_frames=len(software)//25344, hardware_frames=len(hardware)//25344,
                      software_seconds=sw_seconds, hardware_seconds=time.monotonic()-start,
                      sha256=hashlib.sha256(hardware).hexdigest())
        print(json.dumps(result), flush=True)
        assert software and hardware == software, 'Decoded pixel/frame mismatch'


if __name__ == '__main__':
    main()
