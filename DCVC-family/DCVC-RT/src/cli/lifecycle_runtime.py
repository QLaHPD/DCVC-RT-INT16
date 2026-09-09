from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional, Set, Tuple

from src.cli.channel_lifecycle import (
    ChannelValidation,
    CleanupResult,
    cleanup_channel,
    mark_channel_kept,
    record_lifecycle_status,
    validate_channel,
)


@dataclass
class ChannelRuntimeState:
    channel_id: str
    state_path: Path
    total: int
    already_done: int
    pending: int
    processed: int = 0
    failures: int = 0
    active: Set[str] = field(default_factory=set)
    terminal: Set[str] = field(default_factory=set)
    status: str = "queued"
    validation: Optional[ChannelValidation] = None
    cleanup: Optional[CleanupResult] = None
    last_error: str = ""

    def snapshot(self) -> Dict:
        validation = self.validation
        return {
            "channel_id": self.channel_id,
            "total": self.total,
            "already_done": self.already_done,
            "pending": self.pending,
            "processed": self.processed,
            "failures": self.failures,
            "active": len(self.active),
            "status": self.status,
            "reclaimable_bytes": validation.reclaimable_bytes if validation else 0,
            "retained_bytes": validation.retained_bytes if validation else 0,
            "validation_errors": len(validation.errors) if validation else 0,
            "last_error": self.last_error,
        }


class ChannelLifecycleController:
    """Tracks channel completion and runs validation/cleanup off the encode loop."""

    def __init__(self, channels: List[ChannelRuntimeState], policy: str, dry_run: bool = False):
        if policy not in {"prompt", "auto", "keep"}:
            raise ValueError(f"invalid cleanup policy: {policy}")
        self.channels = {channel.channel_id: channel for channel in channels}
        self.policy = policy
        self.dry_run = dry_run
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dcvc-cleanup")
        self._futures: Dict[Future, Tuple[str, str]] = {}
        self._events: Deque[str] = deque(maxlen=200)
        self._closed = False

        for channel in channels:
            if channel.pending == 0:
                self._channel_encoding_terminal(channel)

    def _event(self, text: str):
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        self._events.append(f"[{timestamp}] {text}")

    def events(self) -> List[str]:
        return list(self._events)

    def snapshots(self) -> List[Dict]:
        return [self.channels[channel_id].snapshot() for channel_id in sorted(self.channels)]

    def mark_started(self, channel_id: str, base: str):
        channel = self.channels.get(channel_id)
        if channel is None:
            return
        channel.active.add(base)
        channel.status = "encoding"

    def mark_terminal(self, channel_id: str, base: str, success: bool, error: str = ""):
        channel = self.channels.get(channel_id)
        if channel is None or base in channel.terminal:
            return
        channel.active.discard(base)
        channel.terminal.add(base)
        channel.processed += 1
        if not success:
            channel.failures += 1
            channel.last_error = error
        if channel.processed >= channel.pending:
            self._channel_encoding_terminal(channel)

    def _channel_encoding_terminal(self, channel: ChannelRuntimeState):
        if channel.failures:
            channel.status = "encode-failed"
            record_lifecycle_status(
                channel.state_path,
                "encode-failed",
                "channel-encode-failed",
                failures=channel.failures,
            )
            self._event(f"{channel.channel_id}: encoding finished with {channel.failures} failure(s); originals retained")
            return
        if self.policy == "keep":
            channel.status = "kept"
            mark_channel_kept(channel.state_path, actor="policy:keep")
            self._event(f"{channel.channel_id}: encoding complete; originals kept by policy")
            return
        self._submit_validation(channel)

    def _submit_validation(self, channel: ChannelRuntimeState):
        if channel.status in {"validating", "deleting"}:
            return
        channel.status = "validating"
        future = self._executor.submit(validate_channel, channel.state_path)
        self._futures[future] = (channel.channel_id, "validate")
        self._event(f"{channel.channel_id}: validating retained artifacts")

    def _submit_cleanup(self, channel: ChannelRuntimeState, actor: str):
        if channel.status == "deleting":
            return
        channel.status = "deleting"
        record_lifecycle_status(
            channel.state_path,
            "deleting" if not self.dry_run else "dry-run",
            "cleanup-requested",
            actor=actor,
            dry_run=self.dry_run,
        )
        future = self._executor.submit(cleanup_channel, channel.state_path, self.dry_run, actor)
        self._futures[future] = (channel.channel_id, "cleanup")
        if self.dry_run:
            action = "dry-run cleanup started"
        else:
            action = "exact-path cleanup started; revalidating artifacts before deletion"
        self._event(f"{channel.channel_id}: {action}")

    def poll(self):
        completed = [future for future in self._futures if future.done()]
        for future in completed:
            channel_id, operation = self._futures.pop(future)
            channel = self.channels[channel_id]
            try:
                result = future.result()
            except Exception as exc:
                channel.status = f"{operation}-failed"
                channel.last_error = str(exc)
                self._event(f"{channel_id}: {operation} failed: {exc}")
                continue

            if operation == "validate":
                channel.validation = result
                if not result.eligible:
                    channel.status = "validation-failed"
                    channel.last_error = result.errors[0] if result.errors else "validation failed"
                    record_lifecycle_status(
                        channel.state_path,
                        "validation-failed",
                        "channel-validation-failed",
                        errors=result.errors,
                    )
                    self._event(f"{channel_id}: validation failed; originals retained")
                elif self.policy == "auto":
                    record_lifecycle_status(
                        channel.state_path,
                        "validated",
                        "channel-validation-complete",
                        reclaimable_bytes=result.reclaimable_bytes,
                    )
                    self._submit_cleanup(channel, actor="policy:auto")
                else:
                    channel.status = "awaiting-approval"
                    record_lifecycle_status(
                        channel.state_path,
                        "awaiting-approval",
                        "cleanup-awaiting-approval",
                        reclaimable_bytes=result.reclaimable_bytes,
                        files=sum(not item.already_deleted for item in result.items),
                    )
                    self._event(
                        f"{channel_id}: awaiting deletion approval for "
                        f"{sum(not item.already_deleted for item in result.items)} source file(s)"
                    )
            else:
                channel.cleanup = result
                if result.success:
                    channel.status = "dry-run-complete" if result.dry_run else "cleaned"
                    self._event(
                        f"{channel_id}: cleanup complete; deleted {result.deleted_files} file(s), "
                        f"reclaimed {result.reclaimed_bytes} bytes"
                    )
                else:
                    channel.status = "cleanup-failed"
                    channel.last_error = result.errors[0] if result.errors else "cleanup failed"
                    self._event(f"{channel_id}: cleanup failed; remaining originals retained")

    def approve(self, channel_id: str) -> bool:
        channel = self.channels.get(channel_id)
        if channel is None or channel.status != "awaiting-approval":
            return False
        self._submit_cleanup(channel, actor="user:tui")
        return True

    def keep(self, channel_id: str) -> bool:
        channel = self.channels.get(channel_id)
        if channel is None or channel.status != "awaiting-approval":
            return False
        mark_channel_kept(channel.state_path, actor="user:tui")
        channel.status = "kept"
        self._event(f"{channel_id}: originals kept by user")
        return True

    def retry(self, channel_id: str) -> bool:
        channel = self.channels.get(channel_id)
        if channel is None or channel.status not in {"validation-failed", "cleanup-failed", "validate-failed"}:
            return False
        channel.last_error = ""
        self._submit_validation(channel)
        return True

    def awaiting_approvals(self) -> List[str]:
        return [
            channel_id for channel_id, channel in sorted(self.channels.items())
            if channel.status == "awaiting-approval"
        ]

    def awaiting_actions(self) -> List[str]:
        return [
            channel_id for channel_id, channel in sorted(self.channels.items())
            if channel.status in {
                "awaiting-approval", "validation-failed", "cleanup-failed", "validate-failed"
            }
        ]

    def has_background_work(self) -> bool:
        return bool(self._futures)

    def close(self, wait: bool = True):
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=not wait)
