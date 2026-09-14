"""Resumable UF video archives with atomic publication and exact-path cleanup."""

from contextlib import nullcontext
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
from src.archive.dashboard import event_queue
import shutil
import subprocess
import sys
import tempfile
import time

from src.archive.media import (VIDEO_EXTENSIONS, FrameReader, decoder_thread_budget,
                               dimensions, encode_audio, probe, verify_audio)
from src.archive.storage import (SharedWorkPool, Heartbeat, atomic_json, bundle_path, event,
                                 identity, load_bundle, sha256, sync_directory, artifact_path, control_path,
                                 archive_location, archive_lease, publish_flat, recover_flat)


ROOT = Path(__file__).resolve().parents[2]


def model_paths(variant, image=None, video=None):
    return (str(Path(image or ROOT / 'checkpoints/cvpr2026_image.pth.tar').resolve()),
            str(Path(video or ROOT / f'checkpoints/cvpr2026_video_{variant}.pth.tar').resolve()))


def pipeline(args, image_only=False):
    image, video = model_paths(args.model_structure, args.model_path_i, args.model_path_p)
    paths = (image, video)
    runtime = getattr(args, 'runtime', 'fp16')
    if runtime == 'int16':
        from src.int16.prepared import resolve_prepared
        from src.int16.ops import ARITHMETIC_ID
        prepared_i, data_i = resolve_prepared(image, 'image', args.prepared_i)
        prepared_p, data_p = (None,{'source_sha256':None,'identity':'0'*64}) if image_only else resolve_prepared(video, args.model_structure, args.prepared_p)
        hashes = (data_i['source_sha256'], data_p['source_sha256'])
    else:
        hashes = (sha256(image), None if image_only else sha256(video))
    config = {'codec': 'dcvc-uf', 'archive_version': 1, 'runtime': runtime,
            'variant': args.model_structure, 'image_sha256': hashes[0], 'video_sha256': hashes[1],
            'qp_i': args.qp_i, 'qp_p': args.qp_p, 'fps': args.fps, 'resolution': args.resolution,
            'reset_interval': args.reset_interval, 'intra_period': args.intra_period,
            'max_frames': args.max_frames, 'audio': args.audio,
            'opus_channels': args.opus_channels, 'opus_bitrate': args.opus_bitrate}
    if image_only: config['image_only'] = True
    for key,default in (('opus_frame_ms',20),('opus_complexity',10),('opus_vbr','on')):
        value = getattr(args,key,default)
        if value != default: config[key] = value
    if getattr(args, 'skip_thres', 0):
        config['skip_threshold'] = args.skip_thres
    if config['runtime'] == 'int16':
        config['integer'] = {'arithmetic': ARITHMETIC_ID, 'image': data_i['identity'], 'video': data_p['identity']}
        paths += (str(prepared_i), str(prepared_p) if prepared_p is not None else None)
        event('integer_preparation_ready', prepared_i=str(prepared_i), prepared_p=str(prepared_p), **config['integer'])
    return config, paths


def make_codec(config, paths, device):
    extra = {'image_only': True} if config.get('image_only') else {}
    if config.get('runtime', 'fp16') == 'int16':
        from src.int16.codec import IntegerCodec
        from src.int16.ops import ARITHMETIC_ID
        if config['integer']['arithmetic'] != ARITHMETIC_ID:
            raise ValueError('Unsupported integer arithmetic version')
        return IntegerCodec(*paths[:2], variant=config['variant'], device=device,
                            prepared_i=paths[2], prepared_p=paths[3],
                            identities=(config['integer']['image'], config['integer']['video']),
                            source_hashes=(config['image_sha256'], config['video_sha256']),
                            skip_threshold=config.get('skip_threshold', 0), **extra)
    if config.get('runtime', 'fp16') != 'fp16' or device == 'cpu':
        raise ValueError('FP16 decoding requires CUDA; CPU is available for the INT16 reference runtime')
    from src.archive.codec import Codec
    return Codec(*paths[:2], variant=config['variant'], device=device,
                 skip_threshold=config.get('skip_threshold', 0), **extra)


def run_prepare(args):
    from src.int16.prepared import resolve_prepared
    for source, variant, output in zip(model_paths(args.model_structure, args.model_path_i, args.model_path_p),
                                       ('image', args.model_structure), (args.prepared_i, args.prepared_p)):
        path, data = resolve_prepared(source, variant, output)
        event('integer_model_prepared', path=str(path), variant=variant, identity=data['identity'])
    return 0


def discover(args):
    if args.input_file:
        from src.archive.thumbnails import EXTENSIONS
        if Path(args.input_file).suffix.lower() in EXTENSIONS:
            if getattr(args, 'thumbnail_codec', 'keep') != 'dcvc-intra':
                raise ValueError('Image inputs require --thumbnail_codec dcvc-intra')
            return []
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
    input_threads = getattr(args, 'input_threads', None)
    if input_threads is None:
        input_threads = decoder_thread_budget(getattr(args, 'procs', 1)) if integer else 1
    prefetch_frames = getattr(args, 'prefetch_frames', None)
    if prefetch_frames is None:
        prefetch_frames = 8 if integer else 0
    selected = {}
    rank = {'.mkv':0,'.mp4':1,'.webm':2,'.mov':3,'.avi':4}
    for source in sorted(set(sources),key=lambda p:(rank.get(p.suffix.lower(),99),str(p))):
        key = (source.relative_to(base).parent,source.stem)
        if key in selected:
            event('duplicate_source_retained',source=str(source),message=f'selected {selected[key].name}; retained duplicate {source.name}')
        else: selected[key]=source
    sources = selected.values()
    for source in sorted(set(sources)):
        relative = source.relative_to(base)
        legacy = output / relative.parent / (source.name + '.uf')
        stem = source.stem
        final = output / relative.parent / (stem + '.uf.json')
        jobs.append({'source': str(source), 'relative': str(relative), 'final': str(final),
                     'input_threads': input_threads, 'prefetch_frames': prefetch_frames,
                     'hwaccel':getattr(args,'ff_hwaccel','none'), 'lease_seconds':getattr(args,'shared_lease_seconds',900), 'layout': 'flat', 'filename_style': 'rt', 'legacy': str(legacy), 'lease_key': legacy.name})
    return jobs


def _copy_sidecars(source, stage):
    for suffix in ('.info.json', '.description', '.jpg', '.jpeg', '.png', '.webp'):
        sidecar = source.with_suffix(suffix)
        if sidecar.is_file() and not sidecar.is_symlink():
            shutil.copy2(sidecar, stage / sidecar.name)


def encode_job(job, config, paths, device, instance):
    source, final = Path(job['source']), Path(job['final'])
    os.environ['UF_JOB_KEY'] = str(final)
    remote = job.get('remote')
    flat = job.get('layout') == 'flat'
    legacy = Path(job['legacy']) if job.get('legacy') else None
    parent, job_id = final.parent, job.get('lease_key', final.name)
    parent.mkdir(parents=True, exist_ok=True)
    os.environ['UF_PROGRESS_LOG'] = str(parent / '.uf-progress.jsonl')
    pool = SharedWorkPool(lease_seconds=job.get('lease_seconds',900), instance_name=instance)
    pool.ensure_pipeline(parent, config)
    lease, owner = pool.try_acquire(parent, job_id)
    if lease is None:
        event('claimed_by_peer', source=str(source), archive=str(final))
        return 'busy'
    stage = None
    try:
        with Heartbeat(lease) as pulse:
            existing = final if final.exists() else (legacy if legacy and legacy.exists() else None)
            if existing is not None:
                _, previous = load_bundle(existing, config)
                if remote and previous['source'].get('url') != remote['url']:
                    raise ValueError('Existing archive has a different source URL')
                if not remote and previous['source']['sha256'] != sha256(source):
                    raise ValueError('Original changed since the existing archive was produced')
                from src.archive.streaming import finish_thumbnail
                finish_thumbnail(remote,final,config,paths,device,instance)
                event('already_encoded', source=str(source), archive=str(existing))
                return 'resumed'
            before = {} if remote else identity(source)
            source_sha = None if remote else sha256(source)
            if flat:
                recovered = recover_flat(final, config, pulse, job_id, source, source_sha,
                                         remote['url'] if remote else None)
                if recovered is not None:
                    event('encode_completed', source=str(source), archive=str(final), recovered=True,
                          **recovered['video'], video_bytes=recovered['artifacts']['video.bin']['size'])
                    return 'encoded'
            original = remote['media'] if remote else probe(source)
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
            from src.archive.streaming import download_pipe
            from src.archive.rt_managed import managed_module
            DecodeError = managed_module('jetson_decode').DecodeError
            mode = job.get('hwaccel','none')
            for decode_mode in (('jetson','none') if mode=='jetson' else (mode,)):
                try:
                    with (download_pipe(remote, remote['video_format']) if remote else nullcontext(None)) as input_pipe, \
                            FrameReader('pipe:0' if remote else source, width, height, original, fps,
                                     decoder_threads=job.get('input_threads', 1),
                                     prefetch_frames=job.get('prefetch_frames', 0), input_pipe=input_pipe,hwaccel=decode_mode) as reader:
                        result = codec.encode(reader, stage / 'video.bin', config['qp_i'], config['qp_p'],
                                              config['reset_interval'], config['intra_period'], config['max_frames'], progress)
                    break
                except DecodeError as exc:
                    if decode_mode != 'jetson': raise
                    (stage/'video.bin').unlink(missing_ok=True)
                    event('decoder_fallback',source=str(source),error=str(exc),message='Retrying input with software decoding')
            result.update(width=width, height=height, fps=fps)
            if config['audio'] == 'opus' and original['audio']:
                existing_audio = parent / (final.name.removesuffix('.uf.json') + '.opus') if flat else None
                if existing_audio is not None and existing_audio.is_file() and not existing_audio.is_symlink():
                    verify_audio(existing_audio)
                    shutil.copy2(existing_audio, stage / 'audio.opus')
                    event('audio_reused', source=str(existing_audio))
                else:
                    with (download_pipe(remote, remote['audio_format']) if remote else nullcontext(None)) as audio_pipe:
                        encode_audio('pipe:0' if remote else source, stage / 'audio.opus', config['opus_channels'], config['opus_bitrate'], result['frames'] / fps, input_pipe=audio_pipe,frame_ms=config.get('opus_frame_ms',20),complexity=config.get('opus_complexity',10),vbr=config.get('opus_vbr','on'))
                verify_audio(stage / 'audio.opus')
            # Publication checks bytes and ownership, without a second neural pass.
            validation = {'mode': 'artifact-hashes', 'decode_performed': False}
            if not remote and (identity(source) != before or sha256(source) != source_sha):
                raise ValueError('Original changed during encoding; refusing to publish')
            if remote:
                existing_info = parent / (final.name.removesuffix('.uf.json')+'.info.json')
                if flat and existing_info.is_file() and not existing_info.is_symlink():
                    info = json.loads(existing_info.read_text())
                    if info.get('id') != remote['public_metadata']['id']:
                        raise ValueError('Existing info.json belongs to another video')
                    shutil.copy2(existing_info,stage/'source.info.json')
                else:
                    atomic_json(stage / 'source.info.json', remote['public_metadata'])
            else:
                _copy_sidecars(source, stage)
            artifacts = {p.name: {'size': p.stat().st_size, 'sha256': sha256(p)} for p in stage.iterdir()}
            import torch
            metadata = {'format': 'dcvc-uf-archive', 'version': 1, 'pipeline': config,
                        'source': {'path': str(source), 'relative_path': job['relative'],
                                   'sha256': source_sha, **before, 'media': original, **({'url': remote['url'], 'kind': 'youtube'} if remote else {})},
                        'video': result, 'validation': validation, 'artifacts': artifacts,
                        'environment': {'torch': torch.__version__, 'cuda': torch.version.cuda,
                                        'gpu': 'CPU reference' if device == 'cpu' else torch.cuda.get_device_name(device)}}
            if flat and job.get('filename_style'):
                metadata['filename_style'] = job['filename_style']
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
            if flat:
                publish_flat(stage, final, metadata, pulse, job_id)
                shutil.rmtree(stage)
            else:
                stage.rename(final)
            stage = None
            sync_directory(parent)
            from src.archive.streaming import finish_thumbnail
            finish_thumbnail(remote,final,config,paths,device,instance,codec)
            os.environ['UF_JOB_KEY'] = str(final)
            os.environ['UF_PROGRESS_LOG'] = str(parent / '.uf-progress.jsonl')
            event('encode_completed', source=str(source), archive=str(final), **result,
                  video_bytes=metadata['artifacts']['video.bin']['size'])
            return 'encoded'
    except Exception as exc:
        event('encode_failed', source=str(source), error=f'{type(exc).__name__}: {exc}')
        return 'failed'
    finally:
        if stage is not None and not (flat and (parent / ('.' + final.name + '.pending')).exists()):
            shutil.rmtree(stage, ignore_errors=True)
        lease.release()


def _job_entry(connection, *args, event_queue=None):
    from src.archive.dashboard import install
    install(event_queue)
    try:
        connection.send(encode_job(*args))
    except Exception as exc:
        event('worker_failed', error=f'{type(exc).__name__}: {exc}')
        connection.send('failed')
    finally:
        connection.close()


def run_encode(args):
    if getattr(args, 'source_urls', None):
        from src.archive.streaming import run_stream
        return run_stream(args)
    jobs = discover(args)
    from src.archive.thumbnails import discover as discover_images
    args._thumbnail_jobs = discover_images(args) if getattr(args,'thumbnail_codec','keep')=='dcvc-intra' else []
    config, paths = pipeline(args,image_only=bool(args._thumbnail_jobs) and not jobs)
    args._validation_jobs = jobs
    args._cleanup_sources = {str(Path(job['source']).absolute()) for job in jobs}
    event('run_discovered', total=len(jobs)+len(args._thumbnail_jobs))
    if not jobs:
        from src.archive.thumbnails import run_pass
        image_failures = run_pass(args, config, paths)
        from src.archive.lifecycle import finish
        from src.archive.dashboard import command_queue
        finish(args, command_queue(), image_failures)
        return int(bool(image_failures))
    outcomes, failed_channels = execute_jobs(args,jobs,config,paths)
    from src.archive.thumbnails import run_pass
    image_failures = run_pass(args, config, paths)
    failed_channels.update(image_failures)
    event('archive_run_completed', **{key: outcomes.count(key) for key in ('encoded', 'resumed', 'busy', 'failed')})
    from src.archive.lifecycle import finish
    from src.archive.dashboard import command_queue
    finish(args, command_queue(), failed_channels)
    return int('failed' in outcomes or bool(image_failures))


def execute_jobs(args, jobs, config, paths, worker_entry=None):
    worker_entry = worker_entry or _job_entry
    devices = ['cpu'] if getattr(args, 'device', 'cuda') == 'cpu' else (args.cuda_idx or [0])
    if args.procs < 1:
        raise ValueError('--procs must be positive')
    # One fresh worker per archive prevents retained native CUDA graphs from
    # accumulating across unrelated resolutions and releases resources on failure.
    context = multiprocessing.get_context('spawn')
    remaining = iter(jobs)
    exhausted = False
    outcomes = []
    failed_channels = set()
    active = []
    free_devices = [devices[index % len(devices)] for index in range(args.procs)]
    try:
        while not exhausted or active:
            while not exhausted and len(active) < args.procs:
                try:
                    job = next(remaining)
                except StopIteration:
                    exhausted = True
                    break
                device = free_devices.pop(0)
                reader, writer = context.Pipe(duplex=False)
                process = context.Process(target=worker_entry, args=(writer, job, config, paths,
                                          device, args.shared_instance),
                                          kwargs={'event_queue': event_queue()})
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
                    if status in ('failed', 'busy'):
                        failed_channels.add(str(Path(job['final']).parent))
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
    return outcomes, failed_channels


def checked_codec(metadata, args):
    config = metadata['pipeline']
    selected = getattr(args, 'runtime', 'auto')
    if selected != 'auto' and selected != config.get('runtime', 'fp16'):
        raise ValueError('Selected runtime does not match archive runtime')
    paths = model_paths(config['variant'], args.model_path_i, args.model_path_p)
    if config.get('runtime', 'fp16') == 'fp16':
        for path, key in list(zip(paths, ('image_sha256', 'video_sha256')))[:1 if config.get('image_only') else 2]:
            if sha256(path) != config[key]:
                raise ValueError(f'Checkpoint does not match archive: {path}')
    if config.get('runtime') == 'int16':
        paths += (args.prepared_i, args.prepared_p)
    return make_codec(config, paths, 'cpu' if getattr(args, 'device', 'cuda') == 'cpu' else args.cuda_idx)


def run_decode(args, verify_only=False, live=False):
    if live:
        from src.archive.frame_source import run_view
        return run_view(args)
    if getattr(args, 'input_folder', None) or getattr(args,'output_folder',None):
        from src.archive.decode import run_batch
        return run_batch(args)
    if Path(args.input).suffix.lower() == '.dcvci':
        from src.archive.thumbnails import decode_image
        return decode_image(args, verify_only)
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
                if ('audio.opus' in metadata['artifacts']) and target.suffix.lower() != '.yuv':
                    command += ['-i', str(artifact_path(root, metadata, 'audio.opus')), '-map', '0:v:0', '-map', '1:a:0', '-c:a', 'copy']
                if target.suffix.lower() == '.yuv':
                    command += ['-c:v','rawvideo','-pix_fmt','yuv420p','-f','rawvideo',temporary]
                else:
                    command += ['-c:v', 'ffv1', '-f', 'matroska', temporary]
            log = tempfile.TemporaryFile()
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        result = codec.decode(artifact_path(root, metadata, 'video.bin'), video, sink=process.stdin if process else None)
        if process:
            process.stdin.close()
            if process.wait() != 0:
                log.seek(0)
                raise RuntimeError(log.read(8192).decode(errors='replace'))
        expected_pixels = metadata.get('validation', {}).get('decoded_yuv_sha256')
        if expected_pixels and result['decoded_yuv_sha256'] != expected_pixels:
            raise ValueError(f"Decoded pixels differ from archive validation ({metadata['pipeline'].get('runtime', 'fp16')}); check runtime and prepared model identity")
        if ('audio.opus' in metadata['artifacts']):
            verify_audio(artifact_path(root, metadata, 'audio.opus'))
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
    if Path(args.input).suffix.lower() == '.dcvci':
        from src.archive.thumbnails import cleanup_image
        return cleanup_image(args)
    root, metadata = load_bundle(args.input)
    pool = SharedWorkPool(instance_name='cleanup')
    parent, key = archive_lease(root, metadata)
    pool.ensure_pipeline(parent, metadata['pipeline'])
    lease, _ = pool.try_acquire(parent, key)
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
    if metadata['source'].get('kind') == 'youtube':
        raise ValueError('Streamed archives have no stored original video to delete')
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
    atomic_json(control_path(root, metadata, 'cleanup-intent.json'), {'source': str(source),
                'sha256': metadata['source']['sha256'], 'requested_at': time.time()})
    source.unlink()
    sync_directory(source.parent)
    atomic_json(control_path(root, metadata, 'cleanup.json'), {'source': str(source), 'sha256': metadata['source']['sha256'],
                                       'deleted_at': time.time()})
    event('original_deleted', source=str(source), archive=str(root))
    return 0
