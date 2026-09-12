#!/usr/bin/env python3
"""Unified DCVC-UF archive commands with FP16 and INT16 runtimes."""

import argparse
import math
import sys

from src.archive.workflow import run_encode, run_decode, run_cleanup, run_prepare


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('must be positive and finite')
    return value


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    encode = commands.add_parser('encode', help='Encode a video or directory into resumable UF archives')
    sources = encode.add_mutually_exclusive_group(required=True)
    sources.add_argument('--input_file', '--input-file')
    sources.add_argument('--base_root', '--input_folder')
    encode.add_argument('--output_root', required=True)
    encode.add_argument('--channel_ids', nargs='+')
    encode.add_argument('--recursive', action=argparse.BooleanOptionalAction, default=True)
    encode.add_argument('--qi', '--qp_i', dest='qp_i', type=int, choices=range(64), default=36)
    encode.add_argument('--qp', '--qp_p', dest='qp_p', type=int, choices=range(64), default=30)
    encode.add_argument('--fps', type=positive)
    encode.add_argument('--resolution', type=int, help='Short edge; preserve aspect ratio. Default: original size')
    encode.add_argument('--model_structure', choices=('hts', 'htl', 'ld'), default='hts')
    encode.add_argument('--model_path_i')
    encode.add_argument('--model_path_p')
    encode.add_argument('--runtime', choices=('fp16', 'int16'), default='fp16')
    encode.add_argument('--device', choices=('cuda', 'cpu'), default='cuda', help='CPU is the INT16 reference backend')
    encode.add_argument('--prepared_i', help='Prepared integer image model/table file')
    encode.add_argument('--prepared_p', help='Prepared integer video model/table file')
    encode.add_argument('--procs', type=int, default=1)
    encode.add_argument('--input_threads', type=int,
                        help='FFmpeg decoder threads per worker (default: INT16 2, FP16 1)')
    encode.add_argument('--prefetch_frames', type=int,
                        help='Read ahead frames (default: INT16 8, FP16 0); queue capped at 8 MiB or one frame')
    encode.add_argument('--cuda_idx', type=int, nargs='+', default=[0])
    encode.add_argument('--reset_interval', type=int, default=32)
    encode.add_argument('--intra_period', '--force_intra_period', type=int, default=-1)
    encode.add_argument('--max_frames', type=int)
    encode.add_argument('--audio', choices=('opus', 'none'), default='opus')
    encode.add_argument('--opus_channels', choices=('mono', 'stereo'), default='mono')
    encode.add_argument('--opus_bitrate', default='6k')
    encode.add_argument('--shared-work', action='store_true', help='Accepted for RT command compatibility; UF always uses cooperative claims')
    encode.add_argument('--shared-instance')
    encode.add_argument('--ui', choices=('auto', 'plain'), default='auto', help='Timestamped events, also persisted beside archives')
    encode.set_defaults(handler=run_encode)
    prepare = commands.add_parser('prepare-int16', help='Prepare portable integer model weights and entropy tables')
    prepare.add_argument('--model_structure', choices=('hts', 'htl', 'ld'), default='hts')
    prepare.add_argument('--model_path_i')
    prepare.add_argument('--model_path_p')
    prepare.add_argument('--prepared_i', help='Prepared image output path')
    prepare.add_argument('--prepared_p', help='Prepared video output path')
    prepare.set_defaults(handler=run_prepare)
    for command in ('decode', 'verify', 'view', 'cleanup'):
        child = commands.add_parser(command)
        child.add_argument('--input', '--input_file', required=True, help='UF archive directory or its video.bin')
        child.add_argument('--model_path_i')
        child.add_argument('--model_path_p')
        child.add_argument('--cuda_idx', type=int, default=0)
        child.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
        child.add_argument('--runtime', choices=('auto', 'fp16', 'int16'), default='auto')
        child.add_argument('--prepared_i')
        child.add_argument('--prepared_p')
        if command == 'decode':
            child.add_argument('--output_file', required=True, help='Lossless reconstructed Matroska file (.mkv)')
        if command == 'cleanup':
            child.add_argument('--base_root', help='Remap original relative path on another machine')
            child.add_argument('--yes', action='store_true', help='Verify archive and delete only its original video')
        child.set_defaults(handler={'decode': run_decode, 'verify': lambda a: run_decode(a, verify_only=True),
                                   'view': lambda a: run_decode(a, live=True), 'cleanup': run_cleanup}[command])
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command == 'encode':
        if args.input_threads is None:
            args.input_threads = 2 if args.runtime == 'int16' else 1
        if args.prefetch_frames is None:
            args.prefetch_frames = 8 if args.runtime == 'int16' else 0
        if args.device == 'cpu' and args.runtime != 'int16':
            parser.error('--device cpu requires --runtime int16')
        for name in ('resolution', 'max_frames', 'procs', 'input_threads'):
            if getattr(args, name) is not None and getattr(args, name) <= 0:
                parser.error(f'--{name} must be positive')
        if any(index < 0 for index in args.cuda_idx):
            parser.error('--cuda_idx must be nonnegative')
        if args.prefetch_frames < 0:
            parser.error('--prefetch_frames must be nonnegative')
        if args.intra_period == 0 or args.intra_period < -1:
            parser.error('--intra_period must be -1 or positive')
        if args.reset_interval < -1:
            parser.error('--reset_interval must be -1, 0 or positive')
    elif getattr(args, 'cuda_idx', 0) < 0:
        parser.error('--cuda_idx must be nonnegative')
    try:
        return args.handler(args)
    except KeyboardInterrupt:
        print('Interrupted; completed archives and originals are retained.', file=sys.stderr)
        return 130
    except Exception as exc:
        print(f'Error: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
