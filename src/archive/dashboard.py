"""Plain events and an interactive terminal dashboard for UF managed jobs."""
import curses
import json
import multiprocessing
import queue
import sys
import time
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
        self.messages = deque(maxlen=10)
        self.selected = 0
        self.confirm = None
        self.finished = False

    def update(self, record):
        kind = record.get('event', '')
        source = record.get('source')
        if kind == 'lifecycle_inventory':
            self.channels = record['channels']
            self.selected = min(self.selected, max(0, len(self.channels)-1))
            return
        if kind == 'backend_finished':
            self.finished = True
        if source and kind in ('encode_started', 'encode_progress', 'validation_started'):
            self.active.setdefault(source, {}).update(record)
        if source and kind in ('encode_failed', 'original_deleted', 'encode_completed', 'already_encoded'):
            self.active.pop(source, None)
        if kind != 'encode_progress':
            stamp = record.get('time', '')[11:19]
            detail = record.get('error') or record.get('message') or source or record.get('archive') or ''
            self.messages.append(f'{stamp} {kind}: {detail}')

    def key(self, key):
        if self.confirm:
            action = self.confirm if key in (ord('y'), ord('Y')) else None
            self.confirm = None
            return action
        if key in (ord('q'), ord('Q')):
            return {'action': 'quit'}
        if key in (curses.KEY_UP, ord('k')):
            self.selected = max(0, self.selected-1)
        if key in (curses.KEY_DOWN, ord('j')):
            self.selected = min(max(0, len(self.channels)-1), self.selected+1)
        if key in (ord('d'), ord('D')) and self.channels:
            row = self.channels[self.selected]
            if row['originals'] and not row.get('blocked'):
                self.confirm = {'action': 'delete', 'channel': row['channel']}
        if key in (ord('a'), ord('A')) and any(r['originals'] and not r.get('blocked') for r in self.channels):
            self.confirm = {'action': 'auto'}
        return None


def render(screen, state):
    screen.erase()
    height, width = screen.getmaxyx()
    lines = ['DCVC-UF managed encoding | q exit | arrows select | d delete selected | a delete all eligible', '']
    for source, row in list(state.active.items())[-3:]:
        lines.append(f"{source}: {row['event']}  {row.get('frames', 0):,} frames  {row.get('fps', 0):.1f} FPS")
    lines += ['', 'Channel                         Archives   Originals    Reclaim MiB   Status']
    visible = max(1, height - 14)
    start = max(0, state.selected - visible + 1)
    for index in range(start, min(len(state.channels), start + visible)):
        row = state.channels[index]
        label = row['channel'].rstrip('/').split('/')[-1]
        status = 'blocked' if row.get('blocked') else 'review deletion' if row['originals'] else 'retained (ineligible)' if row.get('ineligible') else 'no local originals'
        lines.append(f"{'>' if index == state.selected else ' '} {label[:29]:29} {len(row['archives']):7} {row['originals']:11} {row['bytes']/1048576:13.1f}   {status}")
    lines += ['', *list(state.messages)[-3:]]
    if state.confirm:
        lines.insert(1, 'DELETE verified original video files? y confirms; any other key cancels.')
    if state.finished:
        lines += ['', 'Run finished. Press q to close.']
    for index, line in enumerate(lines[:max(0,height-1)]):
        try:
            screen.addnstr(index, 0, line, max(0,width-1))
        except curses.error:
            pass
    screen.refresh()


def _backend(args, events, commands):
    install(events, commands)
    from src.archive.storage import event
    try:
        code = args.handler(args)
    except BaseException as exc:
        event('backend_error', error=f'{type(exc).__name__}: {exc}')
        code = 1
    event('backend_finished', code=code or 0)


def run(args):
    mode = getattr(args, 'ui', 'plain')
    tui = mode == 'tui' or mode == 'auto' and sys.stdin.isatty() and sys.stdout.isatty()
    if not tui:
        return args.handler(args)
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise ValueError('--ui tui requires an interactive terminal; use --ui plain')
    context = multiprocessing.get_context('spawn')
    events, commands = context.Queue(), context.Queue()
    worker = context.Process(target=_backend, args=(args, events, commands))
    worker.start()
    state = State()
    code = [0]
    def loop(screen):
        screen.timeout(150)
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        while True:
            try:
                while True:
                    record = events.get_nowait()
                    if record.get('event') == 'backend_finished':
                        code[0] = record.get('code', 0)
                    state.update(record)
            except queue.Empty:
                pass
            if not worker.is_alive():
                state.finished = True
            render(screen, state)
            if state.finished and any(getattr(args, name, False) for name in ('auto_delete', 'keep_originals', 'cleanup_dry_run')):
                break
            action = state.key(screen.getch())
            if action:
                commands.put(action)
                if action['action'] == 'quit':
                    break
    try:
        curses.wrapper(loop)
    finally:
        commands.put({'action': 'quit'})
        worker.join(timeout=2)
        if worker.is_alive():
            # Interrupt the supervisor so its finally blocks stop owned workers.
            import os, signal
            os.kill(worker.pid, signal.SIGINT)
            worker.join(timeout=10)
        if worker.is_alive():
            worker.terminate(); worker.join()
        events.close(); commands.close()
    return code[0]
