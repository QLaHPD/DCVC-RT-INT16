"""Adapt UF codec events to the exact RT managed plain/TUI dashboard."""
import multiprocessing
import queue
import sys
import time
from pathlib import Path
from collections import deque

_EVENTS = None
_COMMANDS = None


def install(events=None, commands=None):
    global _EVENTS, _COMMANDS
    _EVENTS, _COMMANDS = events, commands


def event_queue():
    return _EVENTS


def command_queue():
    return _COMMANDS


class State:
    def __init__(self):
        self.channels = []
        self.active = {}
        self.messages = deque(maxlen=40)
        self.finished = False
        self.code = 0
        self.total = 0
        self.done = 0
        self.slots = {}
        self.started = {}

    def update(self, record, dashboard):
        kind = record.get('event', '')
        key = record.get('job_key') or record.get('source')
        if kind == 'lifecycle_inventory':
            self.channels = record['channels']
        elif kind == 'run_discovered':
            self.total = record['total']
            dashboard.total = self.total
            if dashboard._plain is not None:
                dashboard._plain.total = self.total
                dashboard._plain.global_bar.total = self.total
                dashboard._plain._refresh_global()
        elif kind == 'backend_finished':
            self.finished, self.code = True, record.get('code', 0)
        elif key and kind in ('encode_started', 'encode_progress', 'validation_started'):
            if key not in self.slots:
                used = set(self.slots.values())
                self.slots[key] = next(i for i in range(len(used)+1) if i not in used)
                self.started[key] = time.monotonic()
            self.active.setdefault(key, {}).update(record)
            row = self.active[key]
            name = Path(row.get('source', key)).name
            dashboard.set_worker_text(self.slots[key], f"{name}: {'validating' if kind == 'validation_started' else 'encode'} {row.get('frames',0):,} frames {row.get('fps',0):.2f} FPS")
        if kind in ('encode_completed', 'already_encoded', 'encode_failed'):
            self.done += 1
            if key in self.slots:
                dashboard.clear_worker(self.slots.pop(key))
                self.active.pop(key, None)
            elapsed = time.monotonic() - self.started.pop(key) if key in self.started else None
            dashboard.increment_done(elapsed_seconds=elapsed)
        if kind not in ('encode_progress', 'lifecycle_inventory'):
            stamp = record.get('time', '')[11:19]
            detail = record.get('error') or record.get('message') or record.get('resolution') or record.get('source') or record.get('archive') or ''
            text = f'{stamp} {kind}: {detail}'
            self.messages.append(text)
            if not dashboard.interactive:
                dashboard.write(text)
        rows = []
        for row in self.channels:
            count = len(row['archives'])
            status = ('validation-failed' if row.get('blocked') else 'awaiting-approval' if row['originals']
                      else 'retained' if row.get('ineligible') else 'cleaned' if row.get('absent') else 'complete')
            rows.append(dict(channel_id=row['channel'], status=status, processed=count, pending=0,
                         already_done=0, failures=int(row.get('blocked',False)), active=0, total=count,
                         delete_files=row['originals'], reclaimable_bytes=row['bytes'],last_error=row.get('last_error','')))
        dashboard.update_lifecycle(rows, list(self.messages))


def _backend(args, events, commands):
    install(events, commands if args._ui_interactive else None)
    from src.archive.storage import event
    try:
        code = args.handler(args)
    except BaseException as exc:
        event('backend_error', error=f'{type(exc).__name__}: {exc}')
        code = 1
    event('backend_finished', code=code or 0)


def run(args):
    from src.archive.rt_managed import dashboard_module
    dashboard = dashboard_module().EncodeDashboard(0, getattr(args, 'procs', 1) or 1,
                                                   Path(args.output_root), mode=args.ui)
    args._ui_interactive = dashboard.interactive
    context = multiprocessing.get_context('spawn')
    events, commands = context.Queue(), context.Queue()
    worker = context.Process(target=_backend, args=(args, events, commands))
    state = State()
    worker.start()
    try:
        while True:
            try:
                record = events.get(timeout=.1)
                state.update(record, dashboard)
                while True:
                    state.update(events.get_nowait(), dashboard)
            except queue.Empty:
                pass
            for action in dashboard.poll_actions():
                if action.kind == 'exit-approvals':
                    # Like RT, x retains pending originals and exits after work.
                    commands.put({'action':'quit'})
                elif action.kind in ('approve', 'retry', 'keep'):
                    commands.put({'action': {'approve':'delete','retry':'retry','keep':'keep'}[action.kind],
                                  'channel':action.channel_id})
            if state.finished:
                break
            if not worker.is_alive():
                # Drain the last queued completion before interpreting exit status.
                try:
                    while True:state.update(events.get_nowait(), dashboard)
                except queue.Empty:pass
                if worker.exitcode and not state.finished:state.code=worker.exitcode
                break
    finally:
        commands.put({'action':'quit'})
        worker.join(timeout=2)
        if worker.is_alive():
            import os, signal
            os.kill(worker.pid,signal.SIGINT);worker.join(timeout=10)
        if worker.is_alive():worker.terminate();worker.join()
        dashboard.close();events.close();commands.close()
    return state.code
