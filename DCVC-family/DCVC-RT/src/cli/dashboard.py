from __future__ import annotations

import curses
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from src.cli.progress import AverageCompletionTime, MultiWorkerProgress, fmt_hhmmss


@dataclass(frozen=True)
class DashboardAction:
    kind: str
    channel_id: Optional[str] = None


def format_bytes(value: int) -> str:
    amount = float(max(0, value))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024.0 or unit == "TiB":
            return f"{amount:.1f}{unit}" if unit != "B" else f"{int(amount)}B"
        amount /= 1024.0
    return f"{amount:.1f}TiB"


def _clip(value: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    return value[:width - 3] + "..."


def render_dashboard_lines(
    view: str,
    total: int,
    done: int,
    elapsed: float,
    workers: Dict[int, str],
    channels: Sequence[Dict],
    events: Sequence[str],
    width: int,
    height: int,
    selected: int = 0,
    confirming: Optional[str] = None,
    free_bytes: Optional[int] = None,
    average_seconds: Optional[float] = None,
    eta_seconds: Optional[float] = None,
) -> List[str]:
    width = max(20, width)
    height = max(5, height)
    percent = (100.0 * done / total) if total else 100.0
    status_counts: Dict[str, int] = {}
    for channel in channels:
        status = str(channel.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    active = sum(channel.get("active", 0) for channel in channels)
    pending_approvals = status_counts.get("awaiting-approval", 0)
    failures = sum(channel.get("failures", 0) for channel in channels)
    storage = f" free={format_bytes(free_bytes)}" if free_bytes is not None else ""
    average_text = fmt_hhmmss(average_seconds) if average_seconds is not None else "--:--:--"
    eta_text = fmt_hhmmss(eta_seconds) if eta_seconds is not None else "--:--:--"
    lines = [
        _clip(
            f"DCVC managed encode  {percent:5.1f}% {done}/{total}  elapsed={fmt_hhmmss(elapsed)} "
            f"avg/video={average_text} eta={eta_text}",
            width,
        ),
        _clip(
            f"active={active} failures={failures} approvals={pending_approvals}{storage}",
            width,
        ),
        _clip("Views: [w]orkers [c]hannels [a]pprovals [e]vents  Tab/arrow switch", width),
        _clip(f"View: {view}", width),
    ]

    if confirming:
        lines.append(_clip(f"DELETE {confirming}? Press y to confirm, n/Esc to cancel.", width))

    body_height = max(0, height - len(lines) - 1)
    if view == "workers":
        body = [f"worker {wid}: {workers.get(wid, f'[{wid}] idle')}" for wid in sorted(workers)]
        if not body:
            body = ["No encoder workers are active."]
    elif view == "channels":
        body = []
        for channel in channels:
            body.append(
                f"{channel['channel_id']}  {channel['status']}  "
                f"processed={channel['processed']}/{channel['pending']} "
                f"preexisting={channel['already_done']} failures={channel['failures']}"
            )
        if not body:
            body = ["No channels."]
    elif view == "approvals":
        candidates = [
            channel for channel in channels
            if channel.get("status") in {
                "awaiting-approval", "validation-failed", "cleanup-failed", "validate-failed"
            }
        ]
        body = []
        for index, channel in enumerate(candidates):
            marker = ">" if index == selected else " "
            detail = channel.get("last_error") or ""
            body.append(
                f"{marker} {channel['channel_id']}  {channel['status']}  "
                f"files={channel.get('delete_files', channel['total'])} "
                f"reclaim={format_bytes(channel.get('reclaimable_bytes', 0))} "
                f"{detail}"
            )
        if not body:
            body = ["No cleanup approvals or cleanup failures."]
        body.append("Keys: Up/Down select, d delete, k keep, r retry validation, x leave pending and exit after encoding.")
    else:
        body = list(events[-body_height:]) if events else ["No lifecycle events yet."]

    for value in body[:body_height]:
        lines.append(_clip(value, width))
    while len(lines) < height - 1:
        lines.append("")
    lines.append(_clip("Ctrl-C safely stops encoding; in-progress outputs are discarded atomically.", width))
    return lines[:height]


class EncodeDashboard:
    def __init__(self, total: int, worker_count: int, storage_path: Path, mode: str = "auto"):
        if mode not in {"auto", "tui", "plain"}:
            raise ValueError(f"invalid UI mode: {mode}")
        self.total = total
        self.done = 0
        self.t0 = time.time()
        self.completion_time = AverageCompletionTime(worker_count)
        self.storage_path = Path(storage_path)
        self.workers = {wid: f"[{wid}] idle" for wid in range(worker_count)}
        self.channels: List[Dict] = []
        self.events: List[str] = []
        self.views = ["workers", "channels", "approvals", "events"]
        self.view_index = 0
        self.selected = 0
        self.confirming: Optional[str] = None
        self.exit_approvals_requested = False
        self._screen = None
        self._plain = None
        self._last_plain_status: Dict[str, str] = {}

        tty_available = sys.stdin.isatty() and sys.stdout.isatty() and os.environ.get("TERM", "") != "dumb"
        use_tui = mode == "tui" or (mode == "auto" and tty_available)
        if use_tui:
            try:
                self._screen = curses.initscr()
                curses.noecho()
                curses.cbreak()
                self._screen.keypad(True)
                self._screen.nodelay(True)
                try:
                    curses.curs_set(0)
                except curses.error:
                    pass
            except curses.error:
                self._restore_terminal()
                if mode == "tui":
                    raise
        if self._screen is None:
            self._plain = MultiWorkerProgress(total, worker_count, "Encode")
        self._draw()

    @property
    def interactive(self) -> bool:
        return self._screen is not None

    @property
    def global_bar(self):
        return self

    def write(self, text: str):
        if self._plain is not None:
            self._plain.global_bar.write(text)
        else:
            self.events.append(text)
            self._draw()

    def set_done(self, done: int):
        self.done = done
        if self._plain is not None:
            self._plain.set_done(done)
        self._draw()

    def increment_done(self, step: int = 1, elapsed_seconds: Optional[float] = None):
        self.done += step
        self.completion_time.record(elapsed_seconds)
        if self._plain is not None:
            self._plain.increment_done(step, elapsed_seconds=elapsed_seconds)
        self._draw()

    def set_worker_text(self, wid: int, text: str):
        self.workers[wid] = text
        if self._plain is not None:
            self._plain.set_worker_text(wid, text)
        self._draw()

    def clear_worker(self, wid: int):
        self.set_worker_text(wid, f"[{wid}] idle")

    def update_lifecycle(self, channels: Sequence[Dict], events: Sequence[str]):
        self.channels = list(channels)
        self.events = list(events)
        candidates = self._approval_candidates()
        self.selected = max(0, min(self.selected, max(0, len(candidates) - 1)))
        if self._plain is not None:
            for channel in self.channels:
                channel_id = channel["channel_id"]
                status = channel["status"]
                if self._last_plain_status.get(channel_id) != status and status in {
                    "awaiting-approval", "validation-failed", "cleanup-failed", "cleaned", "dry-run-complete"
                }:
                    self._plain.global_bar.write(f"{channel_id}: {status}")
                self._last_plain_status[channel_id] = status
        self._draw()

    def _approval_candidates(self) -> List[Dict]:
        return [
            channel for channel in self.channels
            if channel.get("status") in {
                "awaiting-approval", "validation-failed", "cleanup-failed", "validate-failed"
            }
        ]

    def poll_actions(self) -> List[DashboardAction]:
        if self._screen is None:
            return []
        actions = []
        while True:
            try:
                key = self._screen.getch()
            except curses.error:
                break
            if key == -1:
                break
            if self.confirming:
                if key in (ord("y"), ord("Y")):
                    actions.append(DashboardAction("approve", self.confirming))
                    self.confirming = None
                elif key in (ord("n"), ord("N"), 27):
                    self.confirming = None
                continue
            if key in (9, curses.KEY_RIGHT):
                self.view_index = (self.view_index + 1) % len(self.views)
            elif key == curses.KEY_LEFT:
                self.view_index = (self.view_index - 1) % len(self.views)
            elif key in (ord("w"), ord("W")):
                self.view_index = 0
            elif key in (ord("c"), ord("C")):
                self.view_index = 1
            elif key in (ord("a"), ord("A")):
                self.view_index = 2
            elif key in (ord("e"), ord("E")):
                self.view_index = 3
            elif key == curses.KEY_UP:
                self.selected = max(0, self.selected - 1)
            elif key == curses.KEY_DOWN:
                self.selected += 1
            elif key in (ord("x"), ord("X")):
                actions.append(DashboardAction("exit-approvals"))
            else:
                candidates = self._approval_candidates()
                if candidates:
                    self.selected = min(self.selected, len(candidates) - 1)
                    selected = candidates[self.selected]
                    channel_id = selected["channel_id"]
                    if key in (ord("d"), ord("D")) and selected["status"] == "awaiting-approval":
                        self.confirming = channel_id
                    elif key in (ord("k"), ord("K")) and selected["status"] == "awaiting-approval":
                        actions.append(DashboardAction("keep", channel_id))
                    elif key in (ord("r"), ord("R")):
                        actions.append(DashboardAction("retry", channel_id))
        self._draw()
        return actions

    def _draw(self):
        if self._screen is None:
            return
        try:
            height, width = self._screen.getmaxyx()
            try:
                free_bytes = shutil.disk_usage(self.storage_path).free
            except OSError:
                free_bytes = None
            lines = render_dashboard_lines(
                view=self.views[self.view_index],
                total=self.total,
                done=self.done,
                elapsed=time.time() - self.t0,
                workers=self.workers,
                channels=self.channels,
                events=self.events,
                width=max(1, width - 1),
                height=height,
                selected=self.selected,
                confirming=self.confirming,
                free_bytes=free_bytes,
                average_seconds=self.completion_time.average_seconds,
                eta_seconds=self.completion_time.eta_seconds(self.total, self.done),
            )
            self._screen.erase()
            for row, line in enumerate(lines):
                if row >= height:
                    break
                try:
                    self._screen.addnstr(row, 0, line, max(0, width - 1))
                except curses.error:
                    pass
            self._screen.refresh()
        except curses.error:
            pass

    def _restore_terminal(self):
        if self._screen is None:
            return
        try:
            self._screen.keypad(False)
            curses.nocbreak()
            curses.echo()
            curses.endwin()
        except curses.error:
            pass
        self._screen = None

    def close(self):
        self._restore_terminal()
        if self._plain is not None:
            self._plain.close()
