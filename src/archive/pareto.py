"""Resumable multi-objective search for UF rate/quality configurations."""

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from types import SimpleNamespace

import numpy as np
from tqdm import tqdm

from src.archive.media import FrameReader, decoder_thread_budget, dimensions, probe
from src.archive.storage import atomic_json, sha256


FORMAT = 'dcvc-uf-pareto-search-v1'


def valid_intra_periods(variant, minimum=1, maximum=60):
    """Return periods the model can represent without splitting temporal chunks."""
    chunk = 1 if variant == 'ld' else 8
    values = [value for value in range(minimum, maximum + 1)
              if value == 1 or value % chunk == 0]
    return [-1, *values]


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
            mutate = {name for name in names if rng.random() < 0.35}
            if not mutate:
                mutate.add(rng.choice(names))
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


def _raw_yuv420(frame):
    return (frame[0].tobytes() + frame[1, ::2, ::2].tobytes() +
            frame[2, ::2, ::2].tobytes())


class MemoryFrameReader:
    def __init__(self, frames, width, height):
        self.frames, self.width, self.height = iter(frames), width, height

    def read(self):
        return next(self.frames, None)


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


def run_pareto_search(args):
    source = Path(args.input_file).resolve()
    if source.is_symlink() or not source.is_file():
        raise ValueError('input_file must be an existing regular file, not a symbolic link')
    output = Path(args.output_json).resolve()
    if output == source:
        raise ValueError('output_json cannot replace the input video')
    original = probe(source)
    width, height = dimensions(original['width'], original['height'], args.resolution)
    threads = args.input_threads or decoder_thread_budget(1)
    domains = {
        'qp_i': list(range(64)),
        'qp_p': list(range(64)),
        'intra_period': valid_intra_periods(args.model_structure),
        'reset_interval': list(range(1, 61)),
        'skip_threshold': sorted(set(float(value) for value in args.skip_thresholds)),
    }
    if not domains['skip_threshold']:
        raise ValueError('At least one skip threshold is required')

    from src.archive.workflow import make_codec, pipeline
    base = SimpleNamespace(model_structure=args.model_structure,
        model_path_i=args.model_path_i, model_path_p=args.model_path_p,
        prepared_i=args.prepared_i, prepared_p=args.prepared_p,
        runtime='int16', qp_i=0, qp_p=0, fps=None, resolution=args.resolution,
        reset_interval=1, intra_period=-1, max_frames=None, audio='none',
        opus_channels='mono', opus_bitrate='6k', skip_thres=0)
    config, paths = pipeline(base)
    settings = {
        'input_file': str(source), 'input_sha256': sha256(source),
        'source': original, 'resolution': f"{original['width']}x{original['height']} -> {width}x{height}",
        'runtime': 'int16', 'model_structure': args.model_structure,
        'integer': config['integer'], 'trials': args.trials,
        'initial_samples': args.initial_samples, 'seed': args.seed,
        'input_threads': threads, 'search_space': domains,
        'objective': {'minimize': 'bitstream_bytes', 'maximize': 'psnr_db'},
        'psnr': 'mean per-frame YUV420 PSNR, 6:1:1 Y:U:V weighting',
        'optimizer': 'balanced categorical Latin sampling, then Pareto-front genetic crossover/mutation',
    }
    created = datetime.now(timezone.utc).isoformat()
    if output.exists():
        if not args.resume:
            raise FileExistsError(f'Output JSON already exists: {output}')
        with output.open() as stream:
            state = json.load(stream)
        if state.get('format') != FORMAT or state.get('settings') != settings:
            raise ValueError('Existing Pareto JSON has different source, model, or search settings')
    else:
        state = _state(settings, created)
        _save(output, state, 0.0)
    if len(state['results']) >= args.trials:
        state['status'] = 'complete'
        _save(output, state, state.get('total_search_seconds', 0))
        print(f'Pareto search already complete: {output}')
        return 0
    state['status'] = 'running'
    _save(output, state, state.get('total_search_seconds', 0))

    print(f'Loading {width}x{height} source frames once for {args.trials} trials...', flush=True)
    frames, references = [], []
    with FrameReader(source, width, height, original, original['fps'],
                     decoder_threads=threads, prefetch_frames=8) as reader:
        while (frame := reader.read()) is not None:
            frames.append(frame)
            references.append(_raw_yuv420(frame))
    if not frames:
        raise ValueError('Input yielded no video frames')

    codec = make_codec(config, paths, args.cuda_idx)
    uniform = uniform_configurations(domains, args.initial_samples, args.seed)
    names = tuple(domains)
    seen = {tuple(item['config'][name] for name in names) for item in state['results']}
    rng = random.Random(args.seed + len(state['results']) * 104729)
    prior_elapsed = state.get('total_search_seconds', 0.0)
    started = time.monotonic()
    progress = tqdm(total=args.trials, initial=len(state['results']), unit='config', dynamic_ncols=True,
                    desc='UF Pareto')
    try:
        while len(state['results']) < args.trials:
            index = len(state['results'])
            if index < args.initial_samples:
                candidate = uniform[index]
                phase = 'uniform'
            else:
                candidate = _adaptive_configuration(domains, state['results'], seen, rng)
                phase = 'genetic'
            key = tuple(candidate[name] for name in names)
            seen.add(key)
            trial_started = time.monotonic()
            fd, temporary = tempfile.mkstemp(prefix='.uf-pareto-', suffix='.bin')
            os.close(fd)
            Path(temporary).unlink()
            try:
                codec.set_skip_threshold(candidate['skip_threshold'])
                encoded = codec.encode(MemoryFrameReader(frames, width, height), temporary,
                                       candidate['qp_i'], candidate['qp_p'],
                                       candidate['reset_interval'], candidate['intra_period'])
                size = Path(temporary).stat().st_size
                sink = PsnrSink(references, width, height)
                decoded = codec.decode(temporary, {**encoded, 'width': width, 'height': height}, sink=sink)
                result = {'trial': index + 1, 'phase': phase, 'status': 'ok',
                          'config': candidate, 'bitstream_bytes': size, **sink.result(),
                          'frames': encoded['frames'], 'packets': encoded['packets'],
                          'encode_seconds': encoded['encode_seconds'],
                          'decode_seconds': decoded['decode_seconds'],
                          'trial_seconds': time.monotonic() - trial_started}
            except Exception as exc:
                result = {'trial': index + 1, 'phase': phase, 'status': 'failed',
                          'config': candidate, 'error': f'{type(exc).__name__}: {exc}',
                          'trial_seconds': time.monotonic() - trial_started}
            finally:
                Path(temporary).unlink(missing_ok=True)
            state['results'].append(result)
            elapsed = prior_elapsed + time.monotonic() - started
            _save(output, state, elapsed)
            progress.update()
            if result['status'] == 'ok':
                progress.set_postfix(bytes=result['bitstream_bytes'],
                                     psnr=f"{result['psnr_db']:.3f}",
                                     frontier=len(state['pareto_frontier_trials']))
            else:
                progress.set_postfix(error=result['error'][:40])
        state['status'] = 'complete'
        _save(output, state, prior_elapsed + time.monotonic() - started)
    except KeyboardInterrupt:
        state['status'] = 'interrupted'
        _save(output, state, prior_elapsed + time.monotonic() - started)
        raise
    finally:
        progress.close()
    print(f'Pareto search complete: {output}', flush=True)
    return 0
