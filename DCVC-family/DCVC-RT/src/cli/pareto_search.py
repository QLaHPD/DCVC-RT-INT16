"""Resumable multi-objective search for RT rate/quality configurations."""

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import io
import subprocess
import argparse
import tempfile
import time
from types import SimpleNamespace

import numpy as np
from tqdm import tqdm

from src.cli.shared_work import file_sha256 as sha256


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)



FORMAT = 'dcvc-rt-pareto-search-v1'


def valid_intra_periods():
    return [-1, *range(1, 61)]


def uniform_configurations(domains, count, seed):
    """Balanced categorical Latin sampling, unique across the joint space."""
    capacity = math.prod(len(values) for values in domains.values())
    if count > capacity:
        raise ValueError(f'{count} initial samples exceed the {capacity} configurations')
    rng = random.Random(seed)
    names = tuple(domains)
    for _ in range(1000):
        columns = []
        for values in domains.values():
            column = [values[index % len(values)] for index in range(count)]
            rng.shuffle(column)
            columns.append(column)
        rows = list(zip(*columns))
        if len(set(rows)) == count:
            return [dict(zip(names, row)) for row in rows]
    raise RuntimeError('Unable to construct a unique balanced initial sample')


def pareto_trials(results):
    """Trial numbers on the nondominated size-minimum/PSNR-maximum frontier."""
    successful = [item for item in results if item.get('status') == 'ok']
    successful.sort(key=lambda item: (item['bitstream_bytes'], -item['psnr_db'], item['trial']))
    frontier, best_psnr = [], -math.inf
    for item in successful:
        if item['psnr_db'] > best_psnr:
            frontier.append(item['trial'])
            best_psnr = item['psnr_db']
    return frontier


def _adaptive_configuration(domains, results, seen, rng):
    successful = [item for item in results if item.get('status') == 'ok']
    frontier_ids = set(pareto_trials(results))
    parents = [item['config'] for item in successful if item['trial'] in frontier_ids]
    names = tuple(domains)
    for _ in range(10000):
        if len(parents) < 2 or rng.random() < 0.15:
            candidate = {name: rng.choice(domains[name]) for name in names}
        else:
            first, second = rng.choice(parents), rng.choice(parents)
            candidate = {name: (first[name] if rng.random() < 0.5 else second[name])
                         for name in names}
            mutate = [name for name in names if rng.random() < 0.35]
            if not mutate:
                mutate.append(rng.choice(names))
            for name in mutate:
                values = domains[name]
                center = values.index(candidate[name])
                step = 0
                while step == 0:
                    step = round(rng.gauss(0, max(1, len(values) / 8)))
                candidate[name] = values[max(0, min(len(values) - 1, center + step))]
        key = tuple(candidate[name] for name in names)
        if key not in seen:
            return candidate
    raise RuntimeError('Unable to find another unique Pareto-search configuration')


class PsnrSink:
    """Frame-average YUV420 PSNR with the conventional 6:1:1 weighting."""
    def __init__(self, references, width, height):
        self.references = references
        self.width, self.height = width, height
        self.index = 0
        self.sums = np.zeros(4, dtype=np.float64)

    @staticmethod
    def _psnr(reference, decoded):
        delta = reference.astype(np.float64) - decoded.astype(np.float64)
        mse = np.mean(delta * delta)
        return 99.9 if mse <= 1e-10 else min(99.9, 10 * math.log10(255 * 255 / mse))

    def write(self, raw):
        if self.index >= len(self.references):
            raise ValueError('Decoder produced more frames than the source reference')
        reference = np.frombuffer(self.references[self.index], dtype=np.uint8)
        decoded = np.frombuffer(raw, dtype=np.uint8)
        if decoded.size != reference.size:
            raise ValueError('Decoded frame size differs from the source reference')
        y_size = self.width * self.height
        uv_size = y_size // 4
        values = [self._psnr(reference[:y_size], decoded[:y_size]),
                  self._psnr(reference[y_size:y_size + uv_size], decoded[y_size:y_size + uv_size]),
                  self._psnr(reference[y_size + uv_size:], decoded[y_size + uv_size:])]
        self.sums += ((6 * values[0] + values[1] + values[2]) / 8, *values)
        self.index += 1

    def result(self):
        if self.index != len(self.references):
            raise ValueError(f'Compared {self.index}/{len(self.references)} frames')
        values = self.sums / self.index
        return {'psnr_db': float(values[0]), 'psnr_y_db': float(values[1]),
                'psnr_u_db': float(values[2]), 'psnr_v_db': float(values[3])}


def _state(settings, created_at):
    return {'format': FORMAT, 'status': 'running', 'created_at': created_at,
            'updated_at': created_at, 'settings': settings, 'completed': 0,
            'total_encode_seconds': 0.0, 'total_search_seconds': 0.0,
            'results': [], 'pareto_frontier_trials': []}


def _save(path, state, elapsed):
    state['completed'] = len(state['results'])
    state['successful'] = sum(item.get('status') == 'ok' for item in state['results'])
    state['failed'] = sum(item.get('status') == 'failed' for item in state['results'])
    state['total_encode_seconds'] = sum(item.get('encode_seconds', 0)
                                        for item in state['results'])
    state['total_search_seconds'] = elapsed
    state['pareto_frontier_trials'] = pareto_trials(state['results'])
    state['updated_at'] = datetime.now(timezone.utc).isoformat()
    atomic_json(path, state)


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def threshold_value(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 16:
        raise argparse.ArgumentTypeError('must be finite and between 0 and 16')
    return value


def configure_parser(parser):
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--output_json', required=True)
    parser.add_argument('--resolution', type=positive_int, default=144)
    parser.add_argument('--trials', type=positive_int, default=2000)
    parser.add_argument('--initial_samples', type=positive_int, default=800)
    parser.add_argument('--seed', type=int, default=20260912)
    parser.add_argument('--cuda_idx', type=int, default=0)
    parser.add_argument('--model_path_i', default='./checkpoints/cvpr2025_image.pth.tar')
    parser.add_argument('--model_path_p', default='./checkpoints/cvpr2025_video.pth.tar')
    parser.add_argument('--skip_thresholds', type=threshold_value, nargs='+',
                        default=[0, .05, .1, .15, .2],
                        help='Search values for the existing encode --force_zero_thres control; 0 disables skipping')
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True)


def load_source(path, resolution):
    """Match the UF search's bicubic YUV420 input, preserving source FPS."""
    from fractions import Fraction
    info = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)]))
    video = next(s for s in info['streams'] if s['codec_type'] == 'video')
    sw, sh = int(video['width']), int(video['height'])
    fps = float(Fraction(video['avg_frame_rate']))
    width = max(2, round(sw * resolution / min(sw, sh) / 2) * 2)
    height = max(2, round(sh * resolution / min(sw, sh) / 2) * 2)
    filters = []
    if (sw, sh) != (width, height):
        filters.append(f'scale={width}:{height}:flags=bicubic')
    filters.append(f'fps={fps}')
    command = ['ffmpeg', '-v', 'error', '-nostdin', '-threads', '4', '-noautorotate',
               '-i', str(path), '-map', '0:v:0', '-an', '-sn', '-dn', '-vf', ','.join(filters),
               '-pix_fmt', 'yuv420p', '-vsync', '0', '-f', 'rawvideo', 'pipe:1']
    raw = subprocess.check_output(command)
    frame_bytes = width * height * 3 // 2
    if not raw or len(raw) % frame_bytes:
        raise ValueError('Source decoding produced empty or truncated frames')
    references = [raw[i:i+frame_bytes] for i in range(0, len(raw), frame_bytes)]
    import hashlib
    return raw, references, {'width': width, 'height': height, 'fps': fps,
        'original_width': sw, 'original_height': sh, 'frames': len(references),
        'raw_yuv420_sha256': hashlib.sha256(raw).hexdigest(), 'resize_filter': 'bicubic'}


class RTTrial:
    """Use the production encoder and decoder, with a preloaded read-only input."""
    def __init__(self, image, video, device, raw, references, media):
        import torch
        from src.cli.encode_workflow import NeuralEncoder, EncoderCfg
        from src.codec.frame_decoder import DecoderModels
        self.torch = torch
        self.cfg = EncoderCfg(resolution=media['height'], ffmpeg_prefetch=8)
        self.encoder = NeuralEncoder(torch.device(f'cuda:{device}'), image, video, self.cfg)
        self.models = DecoderModels(self.encoder.i_net, self.encoder.p_net, self.encoder.device)
        self.raw, self.references, self.media = raw, references, media

    def evaluate(self, config):
        from src.codec.frame_decoder import BitstreamFrameSource
        self.cfg.qp_i = config['qp_i']
        self.cfg.qp_p = config['qp_p']
        self.cfg.force_intra_period = config['intra_period']
        self.cfg.reset_interval = config['reset_interval']
        threshold = config['force_zero_thres'] or None
        for net in (self.encoder.i_net, self.encoder.p_net):
            net.gaussian_encoder.force_zero_thres = threshold
        self.cfg.force_zero_thres = threshold
        width, height = self.media['width'], self.media['height']
        # Temporary production logs and bitstream are removed together, even on errors.
        with tempfile.TemporaryDirectory(prefix='rt-pareto-') as folder:
            path = Path(folder) / 'trial.bin'
            pipe = SimpleNamespace(stdout=io.BytesIO(self.raw), wait=lambda timeout=None: 0)
            self.torch.cuda.synchronize(self.encoder.device)
            started = time.monotonic()
            self.encoder.encode_from_ffmpeg_rawpipe(pipe, width, height, path,
                                                    Path(folder), 'trial')
            self.torch.cuda.synchronize(self.encoder.device)
            encode_seconds = time.monotonic() - started
            started = time.monotonic()
            sink = PsnrSink(self.references, width, height)
            with BitstreamFrameSource(path, self.models) as decoder:
                if decoder.index.frame_count != len(self.references):
                    raise ValueError('Encoded frame count differs from reference')
                while (frame := decoder.read_next('yuv420')) is not None:
                    sink.write(frame.y.tobytes() + frame.uv.tobytes())
            self.torch.cuda.synchronize(self.encoder.device)
            return {**sink.result(), 'bitstream_bytes': path.stat().st_size,
                    'frames': len(self.references), 'packets': len(self.references),
                    'encode_seconds': encode_seconds, 'decode_seconds': time.monotonic() - started}


def run(args):
    if args.initial_samples > args.trials or args.cuda_idx < 0:
        raise ValueError('initial_samples must not exceed trials; cuda_idx must be nonnegative')
    from src.codec.frame_decoder import configure_decode_runtime
    configure_decode_runtime('int16', True)
    source = Path(args.input_file).absolute()
    output = Path(args.output_json).absolute()
    if source.is_symlink() or not source.is_file() or output.is_symlink() or source == output:
        raise ValueError('Use a regular input file and a distinct non-symlink JSON output')
    domains = {'qp_i': list(range(64)), 'qp_p': list(range(64)),
               'intra_period': valid_intra_periods(), 'reset_interval': list(range(1, 61)),
               'force_zero_thres': sorted(set(args.skip_thresholds))}
    uniform = uniform_configurations(domains, args.initial_samples, args.seed)
    image, video = str(Path(args.model_path_i).resolve()), str(Path(args.model_path_p).resolve())
    print(f'Loading source once with short edge {args.resolution}...', flush=True)
    source_hash = sha256(source)
    raw, references, media = load_source(source, args.resolution)
    if sha256(source) != source_hash:
        raise ValueError('Source changed while decoding')
    evaluator = RTTrial(image, video, args.cuda_idx, raw, references, media)
    settings = {'input_file': str(source), 'input_sha256': source_hash, 'media': media,
        'runtime': 'int16', 'codec': 'dcvc-rt', 'audio': 'none', 'trials': args.trials,
        'initial_samples': args.initial_samples, 'seed': args.seed, 'search_space': domains,
        'models': {p: sha256(p) for p in (image, video, image+'.int16prep.pt', video+'.int16prep.pt')},
        'psnr': 'mean per-frame exported YUV420 PSNR, 6:1:1 Y:U:V weighting',
        'objective': {'minimize': 'bitstream_bytes', 'maximize': 'psnr_db'},
        'optimizer': 'balanced categorical Latin sampling then Pareto genetic crossover/mutation',
        'encode_timing': 'production encode loop with preloaded input; includes per-config tuning if any; excludes model loading and decode'}
    state = _state(settings, datetime.now(timezone.utc).isoformat())
    if output.exists():
        if not args.resume:
            raise FileExistsError(output)
        state = json.loads(output.read_text())
        if state.get('format') != FORMAT or state.get('settings') != settings:
            raise ValueError('Existing JSON settings/source/models do not match this run')
    names = tuple(domains)
    seen = {tuple(r['config'][n] for n in names) for r in state['results']}
    prior = state.get('total_search_seconds', 0)
    started = time.monotonic()
    state['status'] = 'running'
    _save(output, state, prior)
    print(f"{media['original_width']}x{media['original_height']} -> {media['width']}x{media['height']}; "
          f"{media['frames']} frames, INT16; results: {output}", flush=True)
    try:
        with tqdm(total=args.trials, initial=len(state['results']), desc='RT Pareto',
                  unit='config', dynamic_ncols=True) as bar:
            while len(state['results']) < args.trials:
                index = len(state['results'])
                phase = 'uniform' if index < args.initial_samples else 'genetic'
                candidate = (uniform[index] if phase == 'uniform' else
                    _adaptive_configuration(domains, state['results'], seen,
                                            random.Random(args.seed + index * 104729)))
                begin = time.monotonic()
                result = {'trial': index+1, 'phase': phase, 'config': candidate}
                try:
                    result.update(evaluator.evaluate(candidate), status='ok')
                except Exception as exc:
                    result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                result['trial_seconds'] = time.monotonic() - begin
                state['results'].append(result)
                seen.add(tuple(candidate[n] for n in names))
                _save(output, state, prior + time.monotonic() - started)
                bar.update()
                bar.set_postfix(frontier=len(state['pareto_frontier_trials']),
                                psnr=round(result.get('psnr_db', 0), 3), failed=state['failed'])
                if result['status'] == 'failed':
                    # Stop on an unvalidated codec error; never spend the remaining
                    # budget feeding a possibly poisoned CUDA context.
                    raise RuntimeError(result['error'])
        state['status'] = 'complete'
    except KeyboardInterrupt:
        state['status'] = 'interrupted'
        raise
    except Exception:
        state['status'] = 'error'
        raise
    finally:
        _save(output, state, prior + time.monotonic() - started)
    return 0
