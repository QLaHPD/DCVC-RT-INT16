#!/usr/bin/env python3
"""Measure local UF encoding with synchronized one/multiple-process runs.

Uses existing checkpoints only. Reported throughput includes FFmpeg input and
bitstream writes, with model initialization and optional decoding timed separately.
"""

import argparse
import hashlib
import importlib.util
import json
import multiprocessing
from pathlib import Path
import resource
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(options, index, processes, barrier, connection):
    try:
        import torch
        if options.native_library:
            spec = importlib.util.spec_from_file_location('uf_int16_cuda', options.native_library)
            native = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(native)
            sys.modules['uf_int16_cuda'] = native
        if options.layers_source:
            import src.int16.layers as layers_backend
            spec = importlib.util.spec_from_file_location('src.int16._benchmark_layers', options.layers_source)
            reference_layers = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reference_layers)
            layers_backend.Layers = reference_layers.Layers
        if options.archive_source:
            import src.archive.codec as archive_backend
            spec = importlib.util.spec_from_file_location('src.archive._benchmark_codec', options.archive_source)
            reference_archive = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reference_archive)
            archive_backend.Codec = reference_archive.Codec
        import src.int16.codec as backend
        from src.archive.media import FrameReader, decoder_thread_budget, dimensions, probe
        import uf_int16_cuda

        if options.model_source:
            spec = importlib.util.spec_from_file_location('src.int16._benchmark_reference', options.model_source)
            reference = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reference)
            backend.IntegerModel = reference.IntegerModel
        source = Path(options.input).resolve()
        media = probe(source)
        width, height = dimensions(media['width'], media['height'], options.resolution)
        output = Path(options.output) / f'p{processes}-w{index}'
        output.mkdir(parents=True, exist_ok=False)
        initialized = time.monotonic()
        codec = backend.IntegerCodec(
            ROOT / 'checkpoints/cvpr2026_image.pth.tar',
            ROOT / f'checkpoints/cvpr2026_video_{options.variant}.pth.tar', options.variant,
            prepared_i=next((ROOT / 'checkpoints').glob('cvpr2026_image.*.int16.pt')),
            prepared_p=next((ROOT / 'checkpoints').glob(f'cvpr2026_video_{options.variant}.*.int16.pt')),
            skip_threshold=options.skip_thres)
        coordinated = (options.gpu_scheduling == 'auto' and processes > 1 and options.variant == 'htl'
                       and not options.model_source and min(width, height) >= 144)
        if coordinated:
            codec.set_gpu_gate(Path(options.output)/'gpu.lock')
        torch.cuda.synchronize(codec.device)
        initialization_seconds = time.monotonic() - initialized
        threads = options.input_threads or decoder_thread_budget(processes)

        def encode(path, frames):
            with FrameReader(source, width, height, media, options.fps,
                             decoder_threads=threads, prefetch_frames=8) as reader:
                return codec.encode(reader, path, options.qp_i, options.qp_p,
                                    options.reset_interval, options.intra_period, frames)

        encode(output / 'warmup.bin', min(17, options.frames))
        runs = []
        for repeat in range(options.repeat):
            barrier.wait(timeout=180)
            started = time.monotonic()
            path = output / f'{repeat}.bin'
            result = encode(path, options.frames)
            finished = time.monotonic()
            result.update(started=started, finished=finished,
                          pipeline_seconds=finished-started,
                          fps=result['frames']/result['encode_seconds'],
                          bitstream_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          bytes=path.stat().st_size)
            runs.append(result)
        assert len({r['bitstream_sha256'] for r in runs}) == 1, 'nondeterministic bitstream'
        validation = None
        if options.verify:
            # Keep verification GPU work outside every worker's encode interval.
            barrier.wait(timeout=180)
            validation_started = time.monotonic()
            validation = codec.decode(output / '0.bin', dict(
                width=width, height=height, frames=runs[0]['frames'], packets=runs[0]['packets']))
            validation.update(started=validation_started, finished=time.monotonic())
        connection.send(dict(worker=index, initialization_seconds=initialization_seconds,
                             peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                             torch_version=torch.__version__, cuda_version=torch.version.cuda,
                             gpu=torch.cuda.get_device_name(codec.device),
                             model_module=type(codec.image).__module__,
                             layers_module=type(codec.image.layers).__module__,
                             archive_module=codec.encode.__module__,
                             kernel_revision=getattr(uf_int16_cuda, 'kernel_revision', None),
                             native_library_sha256=hashlib.sha256(Path(uf_int16_cuda.__file__).read_bytes()).hexdigest(),
                             model_preamble=codec.preamble.hex(),
                             gpu_scheduling='serialized' if coordinated else 'concurrent',
                             input_threads=threads, width=width, height=height,
                             runs=runs, validation=validation))
    except BaseException:
        barrier.abort()
        connection.send(dict(worker=index, error=traceback.format_exc()))
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Local synthetic video')
    parser.add_argument('--output', required=True, help='New directory for results and compatibility fixtures')
    parser.add_argument('--variant', choices=('htl', 'hts', 'ld'), default='htl')
    parser.add_argument('--resolution', type=int, default=144)
    parser.add_argument('--fps', type=float, help='Default: preserve source FPS')
    parser.add_argument('--frames', type=int, default=257)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--procs', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--gpu-scheduling', choices=('auto', 'concurrent'), default='auto')
    parser.add_argument('--input-threads', type=int)
    parser.add_argument('--qp-i', type=int, default=10)
    parser.add_argument('--qp-p', type=int, default=14)
    parser.add_argument('--reset-interval', type=int, default=2)
    parser.add_argument('--intra-period', type=int, default=-1)
    parser.add_argument('--skip-thres', type=float, default=0.2)
    parser.add_argument('--verify', action='store_true', help='Decode in memory and record pixel hashes; never save decoded video')
    parser.add_argument('--archive-source', help='Saved src/archive/codec.py for a complete older verification path')
    parser.add_argument('--model-source', help='Saved src/int16/model.py for comparison against an older implementation')
    parser.add_argument('--layers-source', help='Saved src/int16/layers.py for a complete older implementation')
    parser.add_argument('--native-library', help='Saved uf_int16_cuda extension for a complete older implementation')
    options = parser.parse_args()
    if min(options.frames, options.repeat, *options.procs) < 1:
        parser.error('frames, repeat and procs must be positive')
    output = Path(options.output)
    output.mkdir(parents=True, exist_ok=False)
    model_source = Path(options.model_source) if options.model_source else ROOT / 'src/int16/model.py'
    report = dict(options=vars(options), groups=[],
                  input_sha256=hashlib.sha256(Path(options.input).read_bytes()).hexdigest(),
                  benchmark_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  model_source_sha256=hashlib.sha256(model_source.read_bytes()).hexdigest())
    layers_source = Path(options.layers_source) if options.layers_source else ROOT / 'src/int16/layers.py'
    archive_source = Path(options.archive_source) if options.archive_source else ROOT / 'src/archive/codec.py'
    report['archive_source_sha256'] = hashlib.sha256(archive_source.read_bytes()).hexdigest()
    report['layers_source_sha256'] = hashlib.sha256(layers_source.read_bytes()).hexdigest()
    context = multiprocessing.get_context('spawn')
    for processes in options.procs:
        barrier = context.Barrier(processes)
        active = []
        started = time.monotonic()
        try:
            for index in range(processes):
                reader, writer = context.Pipe(duplex=False)
                process = context.Process(target=worker, args=(options, index, processes, barrier, writer))
                process.start()
                writer.close()
                active.append((process, reader))
            workers = []
            for process, reader in active:
                while not reader.poll(1):
                    if not process.is_alive():
                        raise RuntimeError(f'worker exited with {process.exitcode}')
                value = reader.recv()
                if 'error' in value:
                    raise RuntimeError(value['error'])
                workers.append(value)
            for process, _ in active:
                process.join(timeout=30)
                if process.is_alive() or process.exitcode != 0:
                    raise RuntimeError(f'worker failed to exit cleanly: {process.exitcode}')
        finally:
            for process, reader in active:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=10)
                    if process.is_alive():
                        process.kill()
                        process.join()
                reader.close()
        aggregate = []
        for repeat in range(options.repeat):
            runs = [w['runs'][repeat] for w in workers]
            elapsed = max(r['finished'] for r in runs)-min(r['started'] for r in runs)
            aggregate.append(sum(r['frames'] for r in runs)/elapsed)
        group = dict(processes=processes, workers=workers,
                     aggregate_pipeline_fps=aggregate,
                     median_aggregate_pipeline_fps=statistics.median(aggregate),
                     total_seconds=time.monotonic()-started)
        if options.verify:
            validations = [w['validation'] for w in workers]
            elapsed = max(v['finished'] for v in validations)-min(v['started'] for v in validations)
            group['aggregate_verification_fps'] = sum(v['decoded_frames'] for v in validations)/elapsed
        report['groups'].append(group)
        (output / 'results.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k: v for k, v in group.items() if k != 'workers'}), flush=True)


if __name__ == '__main__':
    main()
