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


def inventory(roots, sources=None):
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
                remote = data['source'].get('kind') == 'youtube'
                in_scope = sources is None or str(source.absolute()) in sources
                eligible = (in_scope and not remote and data['pipeline'].get('max_frames') is None
                            and (not data['source']['media']['audio'] or 'audio.opus' in data['artifacts'])
                            and bool(data.get('validation', {}).get('decoded_yuv_sha256')))
                entry = channels.setdefault(str(channel), dict(channel=str(channel), archives=[], originals=0,
                        bytes=0, streamed=0, absent=0, ineligible=0))
                exists = source.exists() if not remote else False
                entry['archives'].append(dict(input=str(manifest), source=str(source), eligible=eligible,
                                               remote=remote, exists=exists))
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
    return channels


def cleanup_channel(row, args):
    from src.archive.workflow import run_cleanup
    failures = 0
    event('channel_deletion_started', channel=row['channel'], originals=row['originals'], bytes=row['bytes'])
    for item in row['archives']:
        if not item['eligible']:
            continue
        options = SimpleNamespace(input=item['input'], base_root=None, yes=not getattr(args, 'cleanup_dry_run', False),
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
    def refresh():
        rows = inventory(roots, getattr(args, '_cleanup_sources', None))
        for key, row in rows.items():
            row['blocked'] = key in blocked
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
    event('lifecycle_waiting', message='Select a channel and press d to review deletion; q exits')
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
