"""One local batch, per-channel profiles and cooperative peer completion."""

from copy import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

from src.archive.profiles import Profiles, canonical, resolve_selectors
from src.archive.storage import SharedWorkPool, event, identity, load_bundle, sha256


def job_key(job):
    return str(Path(job['final']).absolute())


def existing_archive(job):
    for name in ('final', 'legacy'):
        if job.get(name) and Path(job[name]).exists():
            return Path(job[name])
    return None


def peer_active(job):
    """Read claims without creating or acquiring them; workers arbitrate races."""
    final = Path(job['final'])
    pool = SharedWorkPool(lease_seconds=job.get('lease_seconds', 900))
    claim = pool.claim_path(final.parent, job.get('lease_key', final.name))
    try:
        try:
            stamp = (claim / 'state.json').stat().st_mtime
        except FileNotFoundError:
            stamp = claim.stat().st_mtime
        return max(0, time.time() - stamp) <= pool.lease_seconds
    except FileNotFoundError:
        return False


def validate_completed(job, config):
    """Only committed archives reach here; no neural decoding or model load."""
    try:
        path = existing_archive(job)
        if path is None:
            raise ValueError('Completed archive disappeared; rerun to rediscover the batch')
        _, metadata = load_bundle(path, config)
        if 'video.bin' not in metadata['artifacts'] or metadata['video']['frames'] < 1:
            raise ValueError('Archive has no completed video')
        source_info = metadata['source']
        if (config['audio'] == 'opus' and source_info['media']['audio']
                and 'audio.opus' not in metadata['artifacts']):
            raise ValueError('Completed archive is missing source audio')
        remote = job.get('remote')
        if remote:
            if source_info.get('url') != remote['url']:
                raise ValueError('Existing archive has a different source URL')
        elif source_info.get('url'):
            raise ValueError('Existing streamed archive has URL identity; it cannot verify this local original')
        else:
            source = Path(job['source'])
            if source.exists():
                before = identity(source)
                if source.is_symlink() or sha256(source) != source_info['sha256'] or identity(source) != before:
                    raise ValueError('Original changed since the existing archive was produced')
        event('already_encoded', job_key=job_key(job), source=job['source'], archive=str(path),
              _progress_log=str(Path(job['final']).parent / '.uf-progress.jsonl'))
        return 'resumed'
    except Exception as exc:
        event('encode_failed', job_key=job_key(job), source=job['source'],
              _progress_log=str(Path(job['final']).parent / '.uf-progress.jsonl'),
              error=f'{type(exc).__name__}: {exc}')
        return 'failed'


def check_completed(jobs, config, procs):
    # executor.map keeps result order, while each completed check emits progress.
    with ThreadPoolExecutor(max_workers=procs, thread_name_prefix='uf-resume') as pool:
        return list(pool.map(lambda job: validate_completed(job, config), jobs))


def local_selections(args):
    """Split a root into channel scopes without changing ordinary input-file use."""
    from src.archive.media import VIDEO_EXTENSIONS
    from src.archive.thumbnails import EXTENSIONS
    if args.input_file:
        return [(Path(args.output_root).resolve(), copy(args))]
    base, output = Path(args.base_root).resolve(), Path(args.output_root).resolve()
    if not base.is_dir():
        raise ValueError(f'Input directory does not exist: {base}')
    if args.channel_ids:
        names = resolve_selectors(base, args.channel_ids, args)
    else:
        direct = ((output / '.dcvc-uf-shared-work/pipeline.json').exists()
                  or (output / '.dcvc-uf-image-work/pipeline.json').exists()
                  or any(path.is_file() and (path.suffix.lower() in VIDEO_EXTENSIONS | EXTENSIONS
                                            or path.name.endswith('.uf.json')) for path in base.iterdir()))
        if direct:
            # A channel-folder shortcut keeps canonical CHANNEL_ID/file paths.
            if base == output:
                args.base_root, args.output_root = str(base.parent), str(output.parent)
                args.channel_ids = [base.name]
                selected = copy(args)
                return [(output, selected)]
            return [(output, copy(args))]
        names = [path.name for path in sorted(base.iterdir()) if path.is_dir()
                 and not path.is_symlink() and not path.name.startswith('.') and not path.name.endswith('.uf')]
        if not names:
            return [(output, copy(args))]
    args.channel_ids = names
    result = []
    for name in names:
        selected = copy(args)
        selected.channel_ids = [name]
        result.append((output / name, selected))
    return result


def archived_job(path, metadata, options):
    """Rebuild a local validation job using this mount's relative source path."""
    path = Path(path)
    flat = path.name.endswith('.uf.json')
    final = path if flat else path.parent
    source_info = metadata['source']
    remote = source_info.get('url')
    relative = Path(source_info.get('relative_path') or Path(source_info['path']).name)
    base = Path(options.base_root).resolve() if options.base_root else Path(options.output_root).resolve()
    source = base / relative
    if relative.is_absolute() or not source.resolve().is_relative_to(base):
        raise ValueError(f'Unsafe saved original relative path in {path}')
    job = dict(source=str(source), relative=str(relative), final=str(final),
               lease_key=metadata.get('lease_key', final.name), lease_seconds=options.shared_lease_seconds,
               completed_only=True)
    if remote:
        job['remote'] = {'url': remote}
    return job


def run_local(args):
    from src.archive.workflow import discover, execute_jobs
    from src.archive.thumbnails import discover as discover_images, run_pass, EXTENSIONS
    selections = local_selections(args)
    profiles = Profiles(args)
    groups = []
    args._cleanup_sources, args._validation_jobs = set(), []
    # All profiles, explicit overrides and local model identities are checked
    # before claims, workers or output publication start in any channel.
    for root, selection in selections:
        image_input = selection.input_file and Path(selection.input_file).suffix.lower() in EXTENSIONS
        initial = [] if image_input else discover(selection)
        profile = profiles.load(root, required=args.command == 'join', image_only=not initial)
        options = copy(profile['options'])
        for field in ('base_root', 'output_root', 'channel_ids', 'input_file'):
            setattr(options, field, getattr(selection, field))
        options._cleanup_sources = args._cleanup_sources
        jobs = discover(options)
        for job in jobs:
            job['allow_missing_original'] = True
        options._thumbnail_jobs = discover_images(options) if options.thumbnail_codec == 'dcvc-intra' else []
        existing, pending, seen = [], [], set()
        for job in jobs:
            archive = existing_archive(job)
            if archive:
                existing.append(job)
                seen.add(str(archive.absolute()))
            else:
                pending.append(job)
        for path, metadata in profile['archives']:
            archive = path if path.name.endswith('.uf.json') else path.parent
            if str(archive.absolute()) in seen:
                continue
            existing.append(archived_job(path, metadata, options))
            seen.add(str(archive.absolute()))
        group = {**profile, 'options': options, 'existing': existing, 'pending': pending}
        groups.append(group)
        args._validation_jobs.extend(existing + pending)
        args._cleanup_sources.update(str(Path(job['source']).absolute()) for job in jobs)
        args._cleanup_sources.update(str(path.absolute()) for path, _ in options._thumbnail_jobs)
    args._channel_profiles = {str(group['root']): group['paths'] for group in groups}
    total = sum(len(group['existing']) + len(group['pending']) + len(group['options']._thumbnail_jobs) for group in groups)
    event('run_discovered', total=total)
    outcomes, failed = {}, set()

    def settled(job, status):
        if status == 'busy':
            return
        key = job_key(job)
        outcomes[key] = status
        if status == 'failed':
            failed.add(str(Path(job['final']).parent))
        event('job_settled', job_key=key, source=job['source'], status=status)

    existing_count = sum(len(group['existing']) for group in groups)
    if existing_count:
        event('resume_check_started', total=existing_count,
              message=f'Checking {existing_count:,} existing video archives before encoding remaining videos')
        for group in groups:
            statuses = check_completed(group['existing'], group['config'], args.procs)
            for job, status in zip(group['existing'], statuses):
                settled(job, status)
        event('resume_check_completed', total=existing_count,
              message=f'Existing archive checks complete: {list(outcomes.values()).count("resumed"):,} already encoded')
    # Combine identical profiles across channels; a profile switch releases all
    # model copies before starting the next pool, keeping one global worker cap.
    batches = {}
    for group in groups:
        key = (canonical(group['config']), group['paths'])
        batch = batches.setdefault(key, {**group, 'pending': []})
        batch['pending'].extend(group['pending'])
    while any(batch['pending'] for batch in batches.values()):
        progressed = False
        for batch in batches.values():
            ready, waiting = [], []
            for job in batch['pending']:
                if existing_archive(job):
                    settled(job, validate_completed(job, batch['config']))
                    progressed = True
                elif peer_active(job):
                    waiting.append(job)
                else:
                    ready.append(job)
            batch['pending'] = waiting
            if ready:
                busy = []
                def result(job, status):
                    if status == 'busy':
                        busy.append(job)
                    else:
                        settled(job, status)
                execute_jobs(batch['options'], ready, batch['config'], batch['paths'], job_result=result)
                batch['pending'].extend(busy)
                progressed = progressed or len(busy) < len(ready)
        remaining = sum(len(batch['pending']) for batch in batches.values())
        if remaining and not progressed:
            event('waiting_for_peers', total=remaining,
                  message=f'Waiting for {remaining:,} peer-owned videos; checking every 5 seconds')
            time.sleep(5)
    for group in groups:
        failed.update(run_pass(group['options'], group['config'], group['paths']))
    event('archive_run_completed', **{key: list(outcomes.values()).count(key)
                                     for key in ('encoded', 'resumed', 'failed')}, busy=0)
    from src.archive.lifecycle import finish
    from src.archive.dashboard import command_queue
    finish(args, command_queue(), failed)
    return int('failed' in outcomes.values() or bool(failed))
