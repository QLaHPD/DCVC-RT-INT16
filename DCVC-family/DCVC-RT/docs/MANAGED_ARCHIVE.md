# Managed archive workflow

## Cleanup policies

The encoder tracks every channel independently. A completed channel can wait for cleanup approval without consuming a model worker or stopping other queued channels.

- Default: validate and wait for TUI approval.
- `--auto-delete`: validate and delete exact inventoried source paths.
- `--cleanup-dry-run`: validate and write records without deleting.
- `--keep-originals`: do not schedule cleanup.

`--auto-delete` and `--keep-originals` are mutually exclusive.

## Terminal dashboard

`--ui auto` selects curses in an interactive terminal and ordinary progress output elsewhere.

| Key | Action |
| --- | --- |
| `w`, `c`, `a`, `e` | Workers, channels, approvals, events |
| Tab, Left, Right | Switch views |
| Up, Down | Select an approval |
| `d`, then `y` | Approve validated deletion |
| `k` | Keep originals |
| `r` | Retry failed validation |
| `x` | Leave approvals pending and exit after encoding |

A non-interactive default run persists `awaiting-approval` and exits without deletion. Approve later with:

```bash
python main.py cleanup \
  --output_root /data/encoded \
  --channel_ids CHANNEL_A
```

The interactive cleanup command requires typing the full channel ID. `--auto-delete` makes this command unattended; `--dry-run` validates only.

## Deletion gates

Every inventoried item in a channel must pass before deletion starts:

1. The source is the same regular file discovered at startup, including path, extension, size, modification timestamp, device, and inode.
2. The latest progress state is successful.
3. The DCVC bitstream filename matches the source and parses strictly to EOF.
4. Its frame count matches recorded encoder statistics when available.
5. Required Opus audio probes successfully with positive duration.
6. Required `.info.json` metadata parses as a JSON object.
7. Known thumbnail sidecars are retained, or every inventoried thumbnail has a valid standalone SPS + I-frame `.dcvci` with matching dimensions and QP.
8. No duplicate basename, symlink, traversal, or unexpected-root condition exists.

Cleanup unlinks only exact inventoried paths. In the default thumbnail `keep` mode, this means source containers only. With `--thumbnail_codec dcvc-intra`, validated original image paths are included in the same approval and their `.dcvci` replacements are retained. Cleanup does not use globs, recursive removal, or directory deletion. Metadata, `.bin`, `.dcvci`, `.opus`, manifests, audit records, and progress logs are retained.

## Persistent channel records

```text
.channel-state.json      Exact inventory and lifecycle status
.archive-manifest.json   Artifact sizes and SHA-256 hashes
.cleanup-audit.jsonl     Append-only authorization/deletion events
.progress.jsonl          Resumable encode progress
.thumbnail-progress.jsonl  Resumable thumbnail intra-code progress
```

The manifest is written before deletion. Each unlink has a durable pre-delete intent and completion audit. If interruption occurs after an unlink but before its completion record, the next cleanup command reconciles that intent. A new inventory is refused while cleanup is marked `deleting` or `partial-failure`.

These files contain local paths and archive history. They belong with production data, not in the source repository.
