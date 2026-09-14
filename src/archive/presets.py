"""Select measured UF configurations without changing manual CLI defaults."""
import argparse
import json
import math
from pathlib import Path

BENCHMARK = Path(__file__).resolve().parents[2] / 'benchmarks/uf-int16-htl-144p-pareto.json'
CONFIG_FIELDS = {'qp_i': 'qp_i', 'qp_p': 'qp_p', 'reset_interval': 'reset_interval',
                 'intra_period': 'intra_period', 'skip_thres': 'skip_threshold'}


class ExplicitCodecOption(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        explicit = set(getattr(namespace, '_explicit_codec_options', ()))
        explicit.add(self.dest)
        namespace._explicit_codec_options = explicit
        setattr(namespace, self.dest, values)


def select_trial(results, target, constraints=None):
    if not math.isfinite(target) or target <= 0:
        raise ValueError('--target-psnr must be positive and finite')
    constraints = constraints or {}
    eligible = []
    for row in results:
        if row.get('status') != 'ok':
            continue
        if any(row['config'].get(key) != value for key, value in constraints.items()):
            continue
        if not all(math.isfinite(row[key]) for key in ('psnr_db', 'encode_seconds', 'bitstream_bytes')):
            continue
        if row['psnr_db'] >= target and row['encode_seconds'] > 0 and row['bitstream_bytes'] > 0:
            eligible.append(row)
    if not eligible:
        raise ValueError(f'No measured configuration meets --target-psnr {target:g} with the specified parameters; '
                         'lower the target, relax parameters, or omit --target-psnr for manual encoding')
    return min(eligible, key=lambda row: (row['encode_seconds'], row['bitstream_bytes'], row['trial']))


def apply_target_psnr(args):
    target = getattr(args, 'target_psnr', None)
    if target is None:
        return
    benchmark = json.loads(BENCHMARK.read_text())
    explicit = getattr(args, '_explicit_codec_options', set())
    settings = benchmark['settings']
    dimensions = settings['resolution'].split(' -> ')[-1].split('x')
    required = dict(runtime=settings['runtime'], model_structure=settings['model_structure'],
                    resolution=min(map(int, dimensions)))
    for field, value in required.items():
        if field in explicit and getattr(args, field) != value:
            raise ValueError(f'Bundled --target-psnr benchmark requires --{field} {value}; '
                             'omit --target-psnr to use other settings')
    constraints = {key: getattr(args, field) for field, key in CONFIG_FIELDS.items() if field in explicit}
    trial = select_trial(benchmark['results'], target, constraints)
    for field, value in required.items():
        setattr(args, field, value)
    for field, key in CONFIG_FIELDS.items():
        setattr(args, field, trial['config'][key])
    args.target_psnr_selection = dict(
        target_psnr=target, benchmark=BENCHMARK.name, trial=trial['trial'],
        measured_psnr=trial['psnr_db'], encode_seconds=trial['encode_seconds'],
        bitstream_bytes=trial['bitstream_bytes'], config=trial['config'],
        message=(f"Benchmark trial {trial['trial']}: {trial['psnr_db']:.3f} dB, "
                 f"{trial['encode_seconds']:.3f}s, {trial['bitstream_bytes']} bytes; "
                 f"{trial['config']}. Reference measurements, not a per-video quality guarantee."))
