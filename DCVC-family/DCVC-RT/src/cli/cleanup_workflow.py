from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.cli.channel_lifecycle import cleanup_channel, find_channel_state, validate_channel
from src.cli.dashboard import format_bytes


def configure_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--output_root",
        required=True,
        help="Root containing channel output folders and their persistent lifecycle state.",
    )
    parser.add_argument(
        "--channel_ids",
        nargs="+",
        required=True,
        help="One or more channel folders to validate and clean.",
    )
    parser.add_argument(
        "--auto-delete",
        action="store_true",
        help="Delete validated originals without an interactive channel-ID confirmation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write manifests and audit intended cleanup without deleting source files.",
    )


def run(args) -> int:
    output_root = Path(args.output_root).resolve()
    failures = 0
    for channel_id in args.channel_ids:
        state_path = find_channel_state(output_root, channel_id)
        try:
            validation = validate_channel(state_path)
        except Exception as exc:
            print(f"{channel_id}: cannot validate lifecycle state: {exc}", file=sys.stderr)
            failures += 1
            continue

        source_count = sum(not item.already_deleted for item in validation.items)
        print(
            f"{channel_id}: sources={source_count} "
            f"reclaimable={format_bytes(validation.reclaimable_bytes)} "
            f"retained={format_bytes(validation.retained_bytes)}"
        )
        if not validation.eligible:
            print(f"{channel_id}: validation failed; nothing will be deleted.", file=sys.stderr)
            for error in validation.errors[:20]:
                print(f"  - {error}", file=sys.stderr)
            if len(validation.errors) > 20:
                print(f"  ... and {len(validation.errors) - 20} more", file=sys.stderr)
            failures += 1
            continue

        if not args.auto_delete and not args.dry_run:
            if not sys.stdin.isatty():
                print(
                    f"{channel_id}: interactive confirmation unavailable; originals retained. "
                    "Use --auto-delete after reviewing validation.",
                    file=sys.stderr,
                )
                failures += 1
                continue
            answer = input(f"Type the full channel ID {channel_id!r} to delete these originals: ")
            if answer != channel_id:
                print(f"{channel_id}: confirmation did not match; originals retained.")
                continue

        actor = "cleanup-cli:auto" if args.auto_delete else (
            "cleanup-cli:dry-run" if args.dry_run else "cleanup-cli:user"
        )
        result = cleanup_channel(state_path, dry_run=args.dry_run, actor=actor)
        if result.success:
            if result.dry_run:
                print(f"{channel_id}: dry run complete; no originals deleted.")
            else:
                print(
                    f"{channel_id}: deleted {result.deleted_files} exact source path(s), "
                    f"reclaimed {format_bytes(result.reclaimed_bytes)}."
                )
        else:
            print(f"{channel_id}: cleanup failed; remaining originals retained.", file=sys.stderr)
            for error in result.errors:
                print(f"  - {error}", file=sys.stderr)
            failures += 1
    return 1 if failures else 0
