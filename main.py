#!/usr/bin/env python3
"""Unified DCVC-UF archive commands with FP16 and INT16 runtimes."""

import argparse
import math
import sys

from src.archive.presets import ExplicitCodecOption

from src.archive.workflow import run_encode, run_decode, run_cleanup, run_prepare
from src.archive.pareto import run_pareto_search

def str2bool(value):
    if value.lower() in ("true", "1", "yes"): return True
    if value.lower() in ("false", "0", "no"): return False
    raise argparse.ArgumentTypeError("expected true or false")


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('must be positive and finite')
    return value


def nonnegative(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError('must be nonnegative and finite')
    return value


def skip_threshold(value):
    value = nonnegative(value)
    if value > 16:
        raise argparse.ArgumentTypeError('must be between 0 and 16')
    return value


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    encode = commands.add_parser('encode', aliases=['stream', 'join'],
                                 help='Encode originals, stream remote videos, or join a saved channel folder')
    sources = encode.add_mutually_exclusive_group(required=False)
    sources.add_argument('--input_file', '--input-file')
    sources.add_argument('--base_root', '--input_folder')
    sources.add_argument('--source_urls', nargs='+', help='YouTube video or channel URLs; stream without keeping source videos')
    encode.add_argument('--root', help='Channel storage root (sets local input and output roots; stream sets output only)')
    encode.add_argument('--channels', nargs='+', help='Channel folder IDs or YouTube/Twitch channel URLs')
    encode.add_argument('--youtube_channels', nargs='+', help='YouTube channel IDs or channel_ID.txt video lists')
    encode.add_argument('--twitch_channels', nargs='+', help='Twitch channel names')
    encode.add_argument('--cookies', help='Cookie file; each yt-dlp call uses a private temporary copy')
    encode.add_argument('--ytdlp_bin', '--yt-dlp', default='yt-dlp')
    encode.add_argument('--source-max-height', type=int, default=480)
    encode.add_argument('--max-videos', type=int)
    encode.add_argument('--no-thumbnail', dest='write_thumbnail', action='store_false')
    encode.add_argument('--output_root')
    encode.add_argument('--channel_ids', nargs='+')
    encode.add_argument('--recursive', action=argparse.BooleanOptionalAction, default=True)
    encode.add_argument('--qi', '--qp_i', action=ExplicitCodecOption, dest='qp_i', type=int, choices=range(64), default=36)
    encode.add_argument('--qp', '--qp_p', action=ExplicitCodecOption, dest='qp_p', type=int, choices=range(64), default=30)
    encode.add_argument('--target-psnr', type=positive, help='Select the fastest measured configuration meeting benchmark PSNR; smaller bytes break ties')
    encode.add_argument('--fps', action=ExplicitCodecOption, type=positive)
    encode.add_argument('--resolution', action=ExplicitCodecOption, type=int, help='Short edge; preserve aspect ratio. Default: original size')
    encode.add_argument('--model_structure', action=ExplicitCodecOption, choices=('hts', 'htl', 'ld'), default='hts')
    encode.add_argument('--model_path_i')
    encode.add_argument('--model_path_p')
    encode.add_argument('--runtime', action=ExplicitCodecOption, choices=('fp16', 'int16'), default='fp16')
    encode.add_argument('--cuda', type=str2bool, default=True)
    encode.add_argument('--thumbnail_codec', action=ExplicitCodecOption, choices=('keep','dcvc-intra'), default='keep')
    encode.add_argument('--thumbnail_qp', action=ExplicitCodecOption, type=int, choices=range(64), default=45)
    encode.add_argument('--device', choices=('cuda', 'cpu'), default='cuda', help='CPU is the INT16 reference backend')
    encode.add_argument('--prepared_i', help='Prepared integer image model/table file')
    encode.add_argument('--prepared_p', help='Prepared integer video model/table file')
    encode.add_argument('--procs', type=int, default=1)
    encode.add_argument('--ff_hwaccel', choices=('none','auto','jetson'), default='none')
    encode.add_argument('--input_threads', type=int,
                        help='FFmpeg decoder threads per worker (default: INT16 CPU budget, up to 4; FP16 1)')
    encode.add_argument('--prefetch_frames', '--ffmpeg_prefetch', type=int,
                        help='Read ahead frames (default: INT16 8, FP16 0); queue capped at 8 MiB or one frame')
    encode.add_argument('--cuda_idx', type=int, nargs='+', default=[0])
    encode.add_argument('--reset_interval', action=ExplicitCodecOption, type=int, default=32)
    encode.add_argument('--intra_period', '--force_intra_period', action=ExplicitCodecOption, type=int, default=-1)
    encode.add_argument('--skip_thres', action=ExplicitCodecOption, type=skip_threshold, default=0,
                        help='Do not code latent residuals whose predicted scale is at or below this value')
    encode.add_argument('--max_frames', action=ExplicitCodecOption, type=int)
    encode.add_argument('--audio', action=ExplicitCodecOption, choices=('opus', 'none'), default='opus')
    encode.add_argument('--opus_channels', action=ExplicitCodecOption, choices=('mono', 'stereo'), default='mono')
    encode.add_argument('--opus_bitrate', action=ExplicitCodecOption, default='6k')
    encode.add_argument('--opus_frame_ms', action=ExplicitCodecOption, type=float, choices=(2.5,5,10,20,40,60), default=60)
    encode.add_argument('--opus_complexity', action=ExplicitCodecOption, type=int, choices=range(11), default=10)
    encode.add_argument('--opus_vbr', action=ExplicitCodecOption, choices=('on','off','constrained'), default='on')
    encode.add_argument('--shared-work', action='store_true', help='Accepted for RT command compatibility; UF always uses cooperative claims')
    encode.add_argument('--shared-instance')
    encode.add_argument('--shared-lease-seconds', type=int, default=900)
    encode.add_argument('--resume-check', choices=('fast', 'full'), default='fast',
                        help='Completed videos: fast checks metadata and sizes (default); full rereads originals and artifacts for SHA-256')
    encode.add_argument('--allow-missing-metadata', action='store_true', help='Permit cleanup without matching info.json files')
    encode.add_argument('--ui', choices=('auto', 'plain', 'tui'), default='auto', help='Timestamped events, also persisted beside archives')
    policy = encode.add_mutually_exclusive_group()
    policy.add_argument('--auto-delete', action='store_true', help='After encoding, verify archives and delete only their local originals')
    policy.add_argument('--keep-originals', action='store_true', help='Retain originals and exit without waiting for TUI cleanup')
    policy.add_argument('--cleanup-dry-run', action='store_true', help='Preview exact original paths without deleting')
    encode.set_defaults(handler=run_encode)
    from src.archive.lifecycle import run_manage
    manage = commands.add_parser('manage', help='Review existing channel archives and delete verified originals')
    manage.add_argument('--output_root', required=True)
    manage.add_argument('--base_root', help='Local mount path of originals when reviewing another machine’s archives')
    manage.add_argument('--channel_ids', nargs='+')
    manage.add_argument('--allow-missing-metadata', action='store_true')
    manage.add_argument('--ui', choices=('auto', 'plain', 'tui'), default='auto')
    manage.add_argument('--model_path_i')
    manage.add_argument('--model_path_p')
    manage.add_argument('--prepared_i')
    manage.add_argument('--prepared_p')
    manage.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    manage.add_argument('--cuda_idx', type=int, default=0)
    policy = manage.add_mutually_exclusive_group()
    policy.add_argument('--auto-delete', action='store_true')
    policy.add_argument('--keep-originals', action='store_true')
    policy.add_argument('--cleanup-dry-run', action='store_true')
    manage.set_defaults(handler=run_manage)
    pareto = commands.add_parser('pareto-search', help='Find size/PSNR Pareto configurations for one video')
    pareto.add_argument('--input_file', '--input-file', required=True)
    pareto.add_argument('--output_json', required=True)
    pareto.add_argument('--resolution', type=int, default=144)
    pareto.add_argument('--model_structure', choices=('hts', 'htl', 'ld'), default='htl')
    pareto.add_argument('--model_path_i')
    pareto.add_argument('--model_path_p')
    pareto.add_argument('--prepared_i')
    pareto.add_argument('--prepared_p')
    pareto.add_argument('--cuda_idx', type=int, default=0)
    pareto.add_argument('--trials', type=int, default=2000)
    pareto.add_argument('--initial_samples', type=int, default=800)
    pareto.add_argument('--seed', type=int, default=20260912)
    pareto.add_argument('--input_threads', type=int)
    pareto.add_argument('--skip_thresholds', type=skip_threshold, nargs='+',
                        default=(0, 0.05, 0.10, 0.15, 0.20))
    pareto.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True)
    pareto.set_defaults(handler=run_pareto_search)
    prepare = commands.add_parser('prepare-int16', help='Prepare portable integer model weights and entropy tables')
    prepare.add_argument('--model_structure', choices=('hts', 'htl', 'ld'), default='hts')
    prepare.add_argument('--model_path_i')
    prepare.add_argument('--model_path_p')
    prepare.add_argument('--prepared_i', help='Prepared image output path')
    prepare.add_argument('--prepared_p', help='Prepared video output path')
    prepare.set_defaults(handler=run_prepare)
    for command in ('decode', 'verify', 'view', 'cleanup'):
        child = commands.add_parser(command)
        child.add_argument('--input', '--input_file', help='Flat .bin/.uf.json, .dcvci image or legacy UF archive')
        child.add_argument('--cuda', type=str2bool, default=True)
        if command == 'view':
            child.add_argument('bin_path', nargs='?')
            child.add_argument('--recursive', action='store_true')
            child.add_argument('--fps', type=positive)
            child.add_argument('--no-audio', action='store_true', help='Disable archived audio during windowed playback')
            child.add_argument('--decode-ahead', type=int, default=512,
                               help='Maximum decoded frames buffered ahead (default: 512, independent of fps)')
            child.add_argument('--decode-history', type=int, default=512,
                               help='Previous decoded frames retained for backward navigation (default: 512)')
            child.add_argument('--start_frame', type=int, default=0)
            child.add_argument('--max_width', type=int, default=1280)
            child.add_argument('--max_height', type=int, default=720)
        if command == 'decode':
            child.add_argument('--worker','-w',type=int,default=1)
            child.add_argument('--input_folder')
            child.add_argument('--output_folder')
            child.add_argument('--input_kind', choices=('all','videos','images'), default='all')
            child.add_argument('--recursive', action='store_true')
        child.add_argument('--model_path_i')
        child.add_argument('--model_path_p')
        child.add_argument('--cuda_idx', type=int, nargs='+' if command == 'decode' else None, default=[0] if command == 'decode' else 0)
        child.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
        child.add_argument('--runtime', choices=('auto', 'fp16', 'int16'), default='auto')
        child.add_argument('--prepared_i')
        child.add_argument('--prepared_p')
        if command == 'decode':
            child.add_argument('--output_file', help='Lossless reconstructed Matroska (.mkv) or intra image (.png)')
        if command == 'cleanup':
            child.add_argument('--base_root', help='Remap original relative path on another machine')
            child.add_argument('--yes', action='store_true', help='Verify archive and delete only its original video')
        child.set_defaults(handler={'decode': run_decode, 'verify': lambda a: run_decode(a, verify_only=True),
                                   'view': lambda a: run_decode(a, live=True), 'cleanup': run_cleanup}[command])
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args,'cuda',True) is False:
        args.device = 'cpu'
    if args.command == 'view':
        args.input = args.input or args.bin_path
        if args.decode_ahead < 1:
            parser.error('--decode-ahead must be positive')
        if args.decode_history < 0:
            parser.error('--decode-history must be nonnegative')
    if args.command in ('decode','verify','view','cleanup'):
        if not args.input and not getattr(args,'input_folder',None):
            parser.error('provide --input_file, --input_folder, or a viewer path')
        if args.command == 'decode' and not (args.output_file or args.output_folder):
            parser.error('decode requires --output_file or --output_folder')
        if args.command == 'decode' and args.input and args.input_folder:
            parser.error('select one of --input_file and --input_folder')
    if args.command in ('encode', 'stream', 'join'):
        try:
            from src.archive.profiles import normalize_inputs
            normalize_inputs(args)
        except ValueError as exc:
            parser.error(str(exc))
        if args.youtube_channels or (args.twitch_channels and not args.base_root):
            if args.input_file or args.base_root:
                parser.error('remote channel inputs cannot be combined with local input')
            from src.archive.streaming import channel_urls
            args.source_urls = (args.source_urls or []) + channel_urls(args.youtube_channels or [],args.twitch_channels or [])
        elif args.twitch_channels and args.base_root:
            args.channel_ids = (args.channel_ids or []) + args.twitch_channels
        if not (args.input_file or args.base_root or args.source_urls):
            parser.error('provide a local input or remote URLs/channels')
        if args.shared_lease_seconds < 30:
            parser.error('--shared-lease-seconds must be at least 30')
        if args.source_max_height < 0 or args.max_videos is not None and args.max_videos < 1:
            parser.error('source height must be nonnegative and max videos positive')
        # Profile-dependent defaults and target selection are applied separately
        # for each channel, after recovering saved codec settings.
        for name in ('resolution', 'max_frames', 'procs', 'input_threads'):
            if getattr(args, name) is not None and getattr(args, name) <= 0:
                parser.error(f'--{name} must be positive')
        if any(index < 0 for index in args.cuda_idx):
            parser.error('--cuda_idx must be nonnegative')
        if args.prefetch_frames is not None and args.prefetch_frames < 0:
            parser.error('--prefetch_frames must be nonnegative')
        if args.intra_period == 0 or args.intra_period < -1:
            parser.error('--intra_period must be -1 or positive')
        if args.reset_interval < -1:
            parser.error('--reset_interval must be -1, 0 or positive')
    elif args.command == 'pareto-search':
        for name in ('resolution', 'trials', 'initial_samples', 'input_threads'):
            value = getattr(args, name, None)
            if value is not None and value <= 0:
                parser.error(f'--{name} must be positive')
        if args.initial_samples > args.trials:
            parser.error('--initial_samples cannot exceed --trials')
        if args.cuda_idx < 0:
            parser.error('--cuda_idx must be nonnegative')
    elif any(i < 0 for i in (args.cuda_idx if isinstance(getattr(args,'cuda_idx',0),list) else [getattr(args,'cuda_idx',0)])):
        parser.error('--cuda_idx must be nonnegative')
    if args.command == 'decode':
        if args.worker < 1: parser.error('--worker must be positive')
        if not args.input_folder and not args.output_folder:
            args.cuda_idx = args.cuda_idx[0]
    try:
        if args.command in ('encode', 'stream', 'join', 'manage'):
            from src.archive.dashboard import run
            return run(args)
        return args.handler(args)
    except KeyboardInterrupt:
        print('Interrupted; completed archives and originals are retained.', file=sys.stderr)
        return 130
    except Exception as exc:
        print(f'Error: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
