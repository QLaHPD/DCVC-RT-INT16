"""Channel inventory and explicit, verified original-file cleanup for UF."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import time

from src.archive.storage import event, manifest_path


def roots_for(args):
    root = Path(args.output_root).resolve()
    names = getattr(args, 'channel_ids', None)
    if not names:
        return [root]
    roots = [(root / name).resolve() for name in names]
    if any(not p.is_relative_to(root) for p in roots):
        raise ValueError('Channel path escapes output_root')
    return roots


def inventory(roots, sources=None, base_root=None):
    """Cheap UI inventory; cleanup separately verifies hashes and decoded pixels."""
    channels = {}
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        candidates = list(root.rglob('*.uf.json')) + list(root.rglob('manifest.json'))
        for manifest in candidates:
            if any(part.startswith('.') for part in manifest.relative_to(root).parts):
                continue
            try:
                data = json.loads(manifest.read_text())
                if data.get('format') != 'dcvc-uf-archive':
                    continue
                channel = manifest.parent if data.get('layout') == 'flat' else manifest.parent.parent
                # A migrated bundle and flat manifest describe one original.
                key = (str(channel), data['source'].get('url') or data['source']['path'], data['source'].get('sha256'))
                if key in seen:
                    continue
                seen.add(key)
                source = Path(data['source']['path'])
                if base_root and data['source'].get('relative_path'):
                    base = Path(base_root).resolve()
                    source = base / data['source']['relative_path']
                    if not source.resolve().is_relative_to(base):
                        raise ValueError('Original path escapes base_root')
                remote = data['source'].get('kind') == 'youtube'
                in_scope = sources is None or str(source.absolute()) in sources
                eligible = (in_scope and not remote and data['pipeline'].get('max_frames') is None
                            and (not data['source']['media']['audio'] or 'audio.opus' in data['artifacts'])
                            and bool(data.get('validation', {}).get('decoded_yuv_sha256')))
                entry = channels.setdefault(str(channel), dict(channel=str(channel), archives=[], originals=0,
                        bytes=0, streamed=0, absent=0, ineligible=0))
                exists = source.exists() if not remote else False
                entry['archives'].append(dict(input=str(manifest), source=str(source), eligible=eligible,
                                               remote=remote, exists=exists, kind='video',
                                               metadata=source.with_suffix('.info.json').is_file() or any(name.endswith('.info.json') for name in data.get('artifacts',{}))))
                if remote:
                    entry['streamed'] += 1
                elif not exists:
                    entry['absent'] += 1
                elif eligible:
                    entry['originals'] += 1
                    entry['bytes'] += source.stat().st_size
                else:
                    entry['ineligible'] += 1
            except (OSError, ValueError, KeyError, TypeError) as exc:
                event('inventory_error', manifest=str(manifest), error=str(exc))
    from src.archive.thumbnails import read_image
    for root in roots:
        if not root.exists(): continue
        for path in root.rglob('*.dcvci'):
            if any(part.startswith('.') for part in path.relative_to(root).parts): continue
            try:
                data,_ = read_image(path, verify_payload=False)
                source = Path(data['source']['path'])
                if base_root and data['source'].get('relative_path'):
                    base = Path(base_root).resolve()
                    source = base / data['source']['relative_path']
                    if not source.resolve().is_relative_to(base):
                        raise ValueError('Original path escapes base_root')
                eligible = sources is None or str(source.absolute()) in sources
                entry = channels.setdefault(str(path.parent),dict(channel=str(path.parent),archives=[],originals=0,
                           bytes=0,streamed=0,absent=0,ineligible=0))
                exists = source.exists()
                entry['archives'].append(dict(input=str(path),source=str(source),eligible=eligible,remote=False,exists=exists,kind='image'))
                if not exists: entry['absent'] += 1
                elif eligible:
                    entry['originals'] += 1; entry['bytes'] += source.stat().st_size
                else: entry['ineligible'] += 1
            except (OSError,ValueError,KeyError,TypeError) as exc:
                event('inventory_error',manifest=str(path),error=str(exc))
    return channels


def cleanup_channel(row, args):
    from src.archive.workflow import run_cleanup
    failures = 0
    event('channel_deletion_started', channel=row['channel'], originals=row['originals'], bytes=row['bytes'])
    for item in row['archives']:
        if not item['eligible']:
            continue
        options = SimpleNamespace(input=item['input'], base_root=getattr(args,'base_root',None), yes=not getattr(args, 'cleanup_dry_run', False),
                    model_path_i=args.model_path_i, model_path_p=args.model_path_p,
                    prepared_i=args.prepared_i, prepared_p=args.prepared_p, runtime='auto',
                    device=args.device, cuda_idx=args.cuda_idx[0] if isinstance(args.cuda_idx, list) else args.cuda_idx)
        try:
            event('original_validation_started', channel=row['channel'], source=item['source'])
            run_cleanup(options)
        except Exception as exc:
            failures += 1
            event('cleanup_failed', channel=row['channel'], source=item['source'], error=str(exc))
    event('channel_deletion_finished', channel=row['channel'], failed=failures,
          dry_run=getattr(args, 'cleanup_dry_run', False))
    return failures


def finish(args, commands=None, failed_channels=()):
    roots = roots_for(args)
    blocked = {str(Path(p).resolve()) for p in failed_channels}
    kept = set()
    def refresh():
        rows = inventory(roots, getattr(args, '_cleanup_sources', None), getattr(args,'base_root',None))
        for key, row in rows.items():
            reasons = channel_blockers(row,args)
            row['blocked'] = key in blocked or bool(reasons)
            row['last_error'] = '; '.join(reasons)
            if key in kept:
                row['ineligible'] += row['originals']
                row['originals'] = 0
        event('lifecycle_inventory', channels=list(rows.values()))
        return rows
    rows = refresh()
    if getattr(args, 'auto_delete', False) or getattr(args, 'cleanup_dry_run', False):
        for row in rows.values():
            if row['originals'] and not row['blocked']:
                cleanup_channel(row, args)
        rows = refresh()
        return
    if commands is None or getattr(args, 'keep_originals', False):
        for row in rows.values():
            if row['originals']:
                event('cleanup_available', channel=row['channel'], originals=row['originals'],
                      blocked=row['blocked'], message='Use --ui tui to review deletion, or --auto-delete for verified automatic cleanup')
        return
    # Stay open even when every video was already encoded. Peer deletions are
    # reflected by periodically rereading source existence and archive inventory.
    event('lifecycle_waiting', message='Approvals: d reviews deletion, y confirms, k keeps originals, r retries validation, x exits after work')
    while True:
        try:
            command = commands.get(timeout=5)
        except Exception as exc:
            from queue import Empty
            if not isinstance(exc, Empty):
                raise
            rows = refresh()
            continue
        action = command.get('action')
        if action == 'quit':
            return
        if action == 'keep':
            kept.add(command.get('channel')); rows = refresh()
        if action == 'retry':
            channel = command.get('channel')
            if revalidate_channel(channel, args):
                blocked.discard(channel)
            rows = refresh()
        if action in ('delete', 'auto'):
            rows = refresh()
            selected = list(rows.values()) if action == 'auto' else [rows[command['channel']]] if command.get('channel') in rows else []
            for row in selected:
                if row['originals'] and not row['blocked']:
                    cleanup_channel(row, args)
                elif row['blocked']:
                    event('cleanup_blocked', channel=row['channel'], message='This encode run had failed or busy jobs; retain originals')
            rows = refresh()


def run_manage(args):
    from src.archive.dashboard import command_queue
    finish(args, command_queue())
    return 0


def revalidate_channel(channel, args):
    """A retry cannot erase failed-work evidence without checking its outputs."""
    from src.archive.storage import load_bundle
    from src.archive.workflow import run_decode
    jobs = [job for job in getattr(args, '_validation_jobs', []) if str(Path(job['final']).parent)==channel]
    if not jobs:
        event('validation_failed',channel=channel,message='No completed-job evidence; rerun encoding to recover missing work')
        return False
    try:
        for job in jobs:
            final = Path(job['final'])
            if not final.exists() and job.get('legacy'): final = Path(job['legacy'])
            load_bundle(final)
            options = SimpleNamespace(input=str(final),runtime='auto',model_path_i=args.model_path_i,
                       model_path_p=args.model_path_p,prepared_i=args.prepared_i,prepared_p=args.prepared_p,
                       device=args.device,cuda_idx=args.cuda_idx[0] if isinstance(args.cuda_idx,list) else args.cuda_idx)
            run_decode(options,verify_only=True)
        return True
    except Exception as exc:
        event('validation_failed',channel=channel,error=str(exc))
        return False


def channel_blockers(row,args):
    from src.archive.media import VIDEO_EXTENSIONS
    reasons=[]
    videos=[item for item in row['archives'] if item.get('kind')=='video' and not item['remote']]
    if not getattr(args,'allow_missing_metadata',True):
        missing=[Path(item['source']).name for item in videos if item['exists'] and not item.get('metadata')]
        if missing: reasons.append(f"Missing info.json for {len(missing)} original(s)")
    # Retained duplicate containers with the same basename are deliberate. A
    # different unprocessed video or metadata-only job blocks channel cleanup.
    covered={Path(item['source']).stem for item in videos}
    directories={Path(item['source']).parent for item in videos}
    for directory in directories:
        if not directory.is_dir(): continue
        for path in directory.iterdir():
            if path.name.startswith('.'): continue
            if path.suffix.lower() in VIDEO_EXTENSIONS and path.stem not in covered:
                reasons.append('Unprocessed source: '+path.name)
            elif path.name.endswith('.info.json') and not path.name.endswith('.uf.info.json'):
                base=path.name.removesuffix('.info.json')
                if base not in covered: reasons.append('Missing video archive for '+path.name)
    return reasons[:10]
