"""Resumable UF video archives with atomic publication and exact-path cleanup."""

import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from src.archive.media import VIDEO_EXTENSIONS, FrameReader, dimensions, encode_audio, probe, verify_audio
from src.archive.storage import (SharedWorkPool, Heartbeat, atomic_json, bundle_path, event,
                                 identity, load_bundle, sha256, sync_directory)


ROOT = Path(__file__).resolve().parents[2]


def model_paths(variant, image=None, video=None):
    return (str(Path(image or ROOT / 'checkpoints/cvpr2026_image.pth.tar').resolve()),
            str(Path(video or ROOT / f'checkpoints/cvpr2026_video_{variant}.pth.tar').resolve()))


def pipeline(args):
    image, video = model_paths(args.model_structure, args.model_path_i, args.model_path_p)
    paths = (image, video)
    runtime = getattr(args, 'runtime', 'fp16')
    if runtime == 'int16':
        from src.int16.prepared import resolve_prepared
        from src.int16.ops import ARITHMETIC_ID
        prepared_i, data_i = resolve_prepared(image, 'image', args.prepared_i)
        prepared_p, data_p = resolve_prepared(video, args.model_structure, args.prepared_p)
        hashes = (data_i['source_sha256'], data_p['source_sha256'])
    else:
        hashes = (sha256(image), sha256(video))
    config = {'codec': 'dcvc-uf', 'archive_version': 1, 'runtime': runtime,
            'variant': args.model_structure, 'image_sha256': hashes[0], 'video_sha256': hashes[1],
            'qp_i': args.qp_i, 'qp_p': args.qp_p, 'fps': args.fps, 'resolution': args.resolution,
            'reset_interval': args.reset_interval, 'intra_period': args.intra_period,
            'max_frames': args.max_frames, 'audio': args.audio,
            'opus_channels': args.opus_channels, 'opus_bitrate': args.opus_bitrate}
    if config['runtime'] == 'int16':
        config['integer'] = {'arithmetic': ARITHMETIC_ID, 'image': data_i['identity'], 'video': data_p['identity']}
        paths += (str(prepared_i), str(prepared_p))
        event('integer_preparation_ready', prepared_i=str(prepared_i), prepared_p=str(prepared_p), **config['integer'])
    return config, paths


def make_codec(config, paths, device):
    if config.get('runtime', 'fp16') == 'int16':
        from src.int16.codec import IntegerCodec
        from src.int16.ops import ARITHMETIC_ID
        if config['integer']['arithmetic'] != ARITHMETIC_ID:
            raise ValueError('Unsupported integer arithmetic version')
        return IntegerCodec(*paths[:2], variant=config['variant'], device=device,
                            prepared_i=paths[2], prepared_p=paths[3],
                            identities=(config['integer']['image'], config['integer']['video']),
                            source_hashes=(config['image_sha256'], config['video_sha256']))
    if config.get('runtime', 'fp16') != 'fp16' or device == 'cpu':
        raise ValueError('FP16 decoding requires CUDA; CPU is available for the INT16 reference runtime')
    from src.archive.codec import Codec
    return Codec(*paths[:2], variant=config['variant'], device=device)


def run_prepare(args):
    from src.int16.prepared import resolve_prepared
    for source, variant, output in zip(model_paths(args.model_structure, args.model_path_i, args.model_path_p),
                                       ('image', args.model_structure), (args.prepared_i, args.prepared_p)):
        path, data = resolve_prepared(source, variant, output)
        event('integer_model_prepared', path=str(path), variant=variant, identity=data['identity'])
    return 0


def discover(args):
    if args.input_file:
        path = Path(args.input_file).absolute()
        if path.is_symlink() or not path.is_file():
            raise ValueError('input_file must be an existing regular file, not a symbolic link')
        sources, base = [path], path.parent
    else:
        base = Path(args.base_root).resolve()
        roots = [base / name for name in args.channel_ids] if args.channel_ids else [base]
        for root in roots:
            if not root.resolve().is_relative_to(base) or not root.is_dir():
                raise ValueError(f'Invalid channel/input directory: {root}')
        sources = []
        for root in roots:
            candidates = root.rglob('*') if args.recursive else root.iterdir()
            sources += [p for p in candidates if p.is_file() and not p.is_symlink()
                        and p.suffix.lower() in VIDEO_EXTENSIONS
                        and not any(part.startswith('.') or part.endswith('.uf') for part in p.relative_to(base).parts)]
    output = Path(args.output_root).resolve()
    jobs = []
    integer = getattr(args, 'runtime', 'fp16') == 'int16'
    for source in sorted(set(sources)):
        relative = source.relative_to(base)
        final = output / relative.parent / (source.name + '.uf')
        jobs.append({'source': str(source), 'relative': str(relative), 'final': str(final),
                     'input_threads': getattr(args, 'input_threads', 2 if integer else 1),
                     'prefetch_frames': getattr(args, 'prefetch_frames', 8 if integer else 0)})
    return jobs


def _copy_sidecars(source, stage):
    for suffix in ('.info.json', '.description', '.jpg', '.jpeg', '.png', '.webp'):
        sidecar = source.with_suffix(suffix)
        if sidecar.is_file() and not sidecar.is_symlink():
            shutil.copy2(sidecar, stage / sidecar.name)


def encode_job(job, config, paths, device, instance):
    source, final = Path(job['source']), Path(job['final'])
    parent, job_id = final.parent, final.name
    parent.mkdir(parents=True, exist_ok=True)
    os.environ['UF_PROGRESS_LOG'] = str(parent / '.uf-progress.jsonl')
    pool = SharedWorkPool(lease_seconds=900, instance_name=instance)
    pool.ensure_pipeline(parent, config)
    lease, owner = pool.try_acquire(parent, job_id)
    if lease is None:
        event('claimed_by_peer', source=str(source), archive=str(final))
        return 'busy'
    stage = None
    try:
        with Heartbeat(lease) as pulse:
            if final.exists():
                _, previous = load_bundle(final, config)
                if previous['source']['sha256'] != sha256(source):
                    raise ValueError('Original changed since the existing archive was produced')
                event('already_encoded', source=str(source), archive=str(final))
                return 'resumed'
            before = identity(source)
            source_sha = sha256(source)
            original = probe(source)
            width, height = dimensions(original['width'], original['height'], config['resolution'])
            fps = config['fps'] or original['fps']
            if not fps or fps <= 0:
                raise ValueError('Cannot determine a positive frame rate; pass --fps')
            event('encode_started', source=str(source), device=device,
                  input_threads=job.get('input_threads', 1),
                  prefetch_frames=job.get('prefetch_frames', 0),
                  runtime=config.get('runtime', 'fp16'),
                  resolution=f"{original['width']}x{original['height']} -> {width}x{height}", fps=fps,
                  variant=config['variant'], qi=config['qp_i'], qp=config['qp_p'])
            stage = Path(tempfile.mkdtemp(prefix='.' + final.name + '.stage-', dir=parent))
            codec = make_codec(config, paths, device)
            last = [0.0]
            def progress(**values):
                now = time.monotonic()
                if now - last[0] >= 2:
                    event('encode_progress', source=source.name, **values)
                    last[0] = now
                    pulse.check()
            with FrameReader(source, width, height, original, fps,
                             decoder_threads=job.get('input_threads', 1),
                             prefetch_frames=job.get('prefetch_frames', 0)) as reader:
                result = codec.encode(reader, stage / 'video.bin', config['qp_i'], config['qp_p'],
                                      config['reset_interval'], config['intra_period'], config['max_frames'], progress)
            result.update(width=width, height=height, fps=fps)
            if config['audio'] == 'opus' and original['audio']:
                encode_audio(source, stage / 'audio.opus', config['opus_channels'], config['opus_bitrate'], result['frames'] / fps)
                verify_audio(stage / 'audio.opus')
            event('validation_started', source=source.name, frames=result['frames'])
            validation = codec.decode(stage / 'video.bin', result)
            if identity(source) != before or sha256(source) != source_sha:
                raise ValueError('Original changed during encoding; refusing to publish')
            _copy_sidecars(source, stage)
            artifacts = {p.name: {'size': p.stat().st_size, 'sha256': sha256(p)} for p in stage.iterdir()}
            import torch
            metadata = {'format': 'dcvc-uf-archive', 'version': 1, 'pipeline': config,
                        'source': {'path': str(source), 'relative_path': job['relative'],
                                   'sha256': source_sha, **before, 'media': original},
                        'video': result, 'validation': validation, 'artifacts': artifacts,
                        'environment': {'torch': torch.__version__, 'cuda': torch.version.cuda,
                                        'gpu': 'CPU reference' if device == 'cpu' else torch.cuda.get_device_name(device)}}
            atomic_json(stage / 'manifest.json', metadata)
            for path in stage.iterdir():
                with path.open('rb') as stream:
                    os.fsync(stream.fileno())
            sync_directory(stage)
            pulse.check()
            # A completed directory is the atomic commit record. os.rename cannot
            # overwrite a nonempty peer archive. Partial work is always hidden.
            if final.exists():
                raise FileExistsError(final)
            stage.rename(final)
            stage = None
            sync_directory(parent)
            event('encode_completed', archive=str(final), **result,
                  video_bytes=metadata['artifacts']['video.bin']['size'])
            return 'encoded'
    except Exception as exc:
        event('encode_failed', source=str(source), error=f'{type(exc).__name__}: {exc}')
        return 'failed'
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)
        lease.release()


def _job_entry(connection, *args):
    try:
        connection.send(encode_job(*args))
    except Exception as exc:
        event('worker_failed', error=f'{type(exc).__name__}: {exc}')
        connection.send('failed')
    finally:
        connection.close()


def run_encode(args):
    config, paths = pipeline(args)
    jobs = discover(args)
    if not jobs:
        raise ValueError('No input video files found')
    devices = ['cpu'] if getattr(args, 'device', 'cuda') == 'cpu' else (args.cuda_idx or [0])
    if args.procs < 1:
        raise ValueError('--procs must be positive')
    # One fresh worker per archive prevents retained native CUDA graphs from
    # accumulating across unrelated resolutions and releases resources on failure.
    context = multiprocessing.get_context('spawn')
    remaining = list(enumerate(jobs))
    outcomes = []
    active = []
    free_devices = [devices[index % len(devices)] for index in range(args.procs)]
    try:
        while remaining or active:
            while remaining and len(active) < args.procs:
                index, job = remaining.pop(0)
                device = free_devices.pop(0)
                reader, writer = context.Pipe(duplex=False)
                process = context.Process(target=_job_entry, args=(writer, job, config, paths,
                                          device, args.shared_instance))
                process.start()
                writer.close()
                active.append((process, reader, job, device))
            for process, reader, job, device in list(active):
                if not process.is_alive():
                    process.join()
                    try:
                        status = reader.recv() if reader.poll() else 'failed'
                    except EOFError:
                        status = 'failed'
                    if process.exitcode:
                        event('worker_failed', source=job['source'], exitcode=process.exitcode)
                        status = 'failed'
                    outcomes.append(status)
                    reader.close()
                    active.remove((process, reader, job, device))
                    free_devices.append(device)
            time.sleep(0.1)
    finally:
        for process, reader, _, _ in active:
            process.terminate()
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join()
            reader.close()
    event('archive_run_completed', **{key: outcomes.count(key) for key in ('encoded', 'resumed', 'busy', 'failed')})
    return int('failed' in outcomes)


def checked_codec(metadata, args):
    config = metadata['pipeline']
    selected = getattr(args, 'runtime', 'auto')
    if selected != 'auto' and selected != config.get('runtime', 'fp16'):
        raise ValueError('Selected runtime does not match archive runtime')
    paths = model_paths(config['variant'], args.model_path_i, args.model_path_p)
    if config.get('runtime', 'fp16') == 'fp16':
        for path, key in zip(paths, ('image_sha256', 'video_sha256')):
            if sha256(path) != config[key]:
                raise ValueError(f'Checkpoint does not match archive: {path}')
    if config.get('runtime') == 'int16':
        paths += (args.prepared_i, args.prepared_p)
    return make_codec(config, paths, 'cpu' if getattr(args, 'device', 'cuda') == 'cpu' else args.cuda_idx)


def run_decode(args, verify_only=False, live=False):
    root, metadata = load_bundle(args.input)
    codec = checked_codec(metadata, args)
    video = metadata['video']
    process = None
    temporary = None
    log = None
    target = None
    try:
        if not verify_only:
            if live:
                command = ['ffplay', '-v', 'error', '-f', 'rawvideo', '-pixel_format', 'yuv420p',
                           '-video_size', f"{video['width']}x{video['height']}", '-framerate', str(video['fps']),
                           '-i', 'pipe:0', '-autoexit']
            else:
                target = Path(args.output_file).resolve()
                if target.exists():
                    raise FileExistsError(f'Refusing to overwrite: {target}')
                target.parent.mkdir(parents=True, exist_ok=True)
                fd, temporary = tempfile.mkstemp(prefix='.' + target.stem, suffix=target.suffix, dir=target.parent)
                os.close(fd)
                command = ['ffmpeg', '-v', 'error', '-nostdin', '-y', '-f', 'rawvideo', '-pix_fmt', 'yuv420p',
                           '-s', f"{video['width']}x{video['height']}", '-r', str(video['fps']), '-i', 'pipe:0']
                if (root / 'audio.opus').exists():
                    command += ['-i', str(root / 'audio.opus'), '-map', '0:v:0', '-map', '1:a:0', '-c:a', 'copy']
                command += ['-c:v', 'ffv1', '-f', 'matroska', temporary]
            log = tempfile.TemporaryFile()
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        result = codec.decode(root / 'video.bin', video, sink=process.stdin if process else None)
        if process:
            process.stdin.close()
            if process.wait() != 0:
                log.seek(0)
                raise RuntimeError(log.read(8192).decode(errors='replace'))
        if result['decoded_yuv_sha256'] != metadata['validation']['decoded_yuv_sha256']:
            raise ValueError(f"Decoded pixels differ from archive validation ({metadata['pipeline'].get('runtime', 'fp16')}); check runtime and prepared model identity")
        if (root / 'audio.opus').exists():
            verify_audio(root / 'audio.opus')
        if temporary:
            os.link(temporary, target)
            os.unlink(temporary)
            temporary = None
        event('verification_completed' if verify_only else 'decode_completed', archive=str(root), **result)
        return 0
    finally:
        if process and process.poll() is None:
            process.kill()
            process.wait()
        if process and process.stdin:
            process.stdin.close()
        if log:
            log.close()
        if temporary:
            Path(temporary).unlink(missing_ok=True)


def run_cleanup(args):
    root, metadata = load_bundle(args.input)
    pool = SharedWorkPool(instance_name='cleanup')
    pool.ensure_pipeline(root.parent, metadata['pipeline'])
    lease, _ = pool.try_acquire(root.parent, root.name)
    if lease is None:
        raise RuntimeError('Archive is owned by another worker; retry cleanup later')
    try:
        with Heartbeat(lease) as heartbeat:
            return _cleanup_owned(args, root, metadata, heartbeat)
    finally:
        lease.release()


def _cleanup_owned(args, root, metadata, heartbeat):
    if metadata['pipeline'].get('max_frames') is not None:
        raise ValueError('Partial archives cannot authorize original deletion')
    if metadata['source']['media']['audio'] and 'audio.opus' not in metadata['artifacts']:
        raise ValueError('Archive omits source audio; refusing original deletion')
    source = (Path(args.base_root).resolve() / metadata['source']['relative_path']
              if args.base_root else Path(metadata['source']['path']))
    if args.base_root and not source.resolve().is_relative_to(Path(args.base_root).resolve()):
        raise ValueError('Original path escapes base_root')
    if not source.exists():
        event('original_already_absent', source=str(source))
        return 0
    if source.is_symlink() or not source.is_file() or sha256(source) != metadata['source']['sha256']:
        raise ValueError('Original differs from the archived source; refusing deletion')
    if not args.yes:
        event('cleanup_preview', source=str(source), bytes=source.stat().st_size,
              message='Pass --yes to verify decoding and delete only this original video')
        return 0
    before = identity(source)
    run_decode(args, verify_only=True)
    if identity(source) != before or sha256(source) != metadata['source']['sha256']:
        raise ValueError('Source changed during validation')
    heartbeat.check()
    atomic_json(root / 'cleanup-intent.json', {'source': str(source),
                'sha256': metadata['source']['sha256'], 'requested_at': time.time()})
    source.unlink()
    sync_directory(source.parent)
    atomic_json(root / 'cleanup.json', {'source': str(source), 'sha256': metadata['source']['sha256'],
                                       'deleted_at': time.time()})
    event('original_deleted', source=str(source), archive=str(root))
    return 0
