"""Capture real entropy inputs or replay a trusted local capture without a GPU.

Capture: --capture FILE -- [benchmark_synthetic_codec.py arguments]
Replay: --replay FILE [--cpp-extension SO] [--iterations 50] --report JSON
Captures contain pickle data and must come from a trusted source.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import pickle
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--capture', type=Path)
    mode.add_argument('--replay', type=Path)
    parser.add_argument('--cpp-extension', type=Path)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--report', type=Path, required=True)
    args, rest = parser.parse_known_args()
    if args.cpp_extension:
        spec = importlib.util.spec_from_file_location('MLCodec_extensions_cpp', args.cpp_extension)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    if args.capture:
        capture(args, rest)
    else:
        if rest or args.iterations < 1:
            parser.error('replay requires a positive iteration count and no extra arguments')
        replay(args)


def capture(args, rest):
    import torch
    from src.models.entropy_models import EntropyCoder
    records, timings = [], {}

    def measure(name, fn):
        start = time.perf_counter()
        result = fn()
        timings.setdefault(name, []).append(time.perf_counter() - start)
        return result

    old_init, old_add = EntropyCoder.__init__, EntropyCoder.add_cdf
    old_reset, old_get = EntropyCoder.reset, EntropyCoder.get_encoded_stream

    def init(self):
        old_init(self)
        self._capture = dict(cdfs=[], frames=[])
        self._calls = []
        records.append(self._capture)

    def add(self, cdf, sizes, offsets):
        result = old_add(self, cdf, sizes, offsets)
        self._capture['cdfs'].append((cdf.copy(), sizes.copy(), offsets.copy()))
        return result

    def reset(self):
        self._calls = []
        old_reset(self)

    def encode_y(self, symbols, group):
        host = measure('y_transfer_wait', lambda: symbols.cpu().numpy())
        measure('y_native_queue', lambda: self.encoder.encode_y(host, group))
        self._calls.append(('y', host.copy(), group))

    def encode_z(self, symbols, group, offset, size):
        host = measure('z_transfer_wait', lambda: symbols.to(torch.int8).cpu().numpy())
        measure('z_native_queue', lambda: self.encoder.encode_z(host, group, offset, size))
        self._calls.append(('z', host.copy(), group, offset, size))

    def get(self):
        result = measure('native_finish_wait_and_output', lambda: old_get(self))
        self._capture['frames'].append((self.encoder.get_use_two_encoders(), self._calls, result))
        return result

    EntropyCoder.__init__, EntropyCoder.add_cdf = init, add
    EntropyCoder.reset, EntropyCoder.get_encoded_stream = reset, get
    EntropyCoder.encode_y, EntropyCoder.encode_z = encode_y, encode_z
    sys.argv = [sys.argv[0]] + (rest[1:] if rest[:1] == ['--'] else rest)
    from benchmarks.benchmark_synthetic_codec import main as run_codec
    run_codec()
    with args.capture.open('wb') as output:
        pickle.dump(records, output)
    report = {key: dict(calls=len(values), mean_ms=1000*statistics.mean(values),
                       median_ms=1000*statistics.median(values)) for key, values in timings.items()}
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def replay(args):
    from MLCodec_extensions_cpp import RansEncoder
    with args.replay.open('rb') as source:
        records = pickle.load(source)
    encoders = []
    for record in records:
        encoder = RansEncoder()
        for table in record['cdfs']:
            encoder.add_cdf(*table)
        encoders.append(encoder)
    wall, cpu = [], []
    frames = sum(len(record['frames']) for record in records)
    assert frames > 0
    for iteration in range(args.iterations + 3):
        t0, c0 = time.perf_counter(), time.process_time()
        for encoder, record in zip(encoders, records):
            for two, calls, expected in record['frames']:
                encoder.set_use_two_encoders(two)
                encoder.reset()
                for kind, *values in calls:
                    getattr(encoder, 'encode_' + kind)(*values)
                encoder.flush()
                actual = encoder.get_encoded_stream().tobytes()
                assert actual == expected, 'entropy bitstream mismatch'
        elapsed_cpu, elapsed_wall = time.process_time()-c0, time.perf_counter()-t0
        if iteration >= 3:
            wall.append(1000*elapsed_wall/frames)
            cpu.append(1000*elapsed_cpu/frames)
    report = dict(frames=frames, iterations=args.iterations, exact=True,
                  median_wall_ms=statistics.median(wall), median_cpu_ms=statistics.median(cpu))
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
