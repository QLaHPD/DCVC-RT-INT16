# One-time SRF Tagesschau archive

`src.archive.srf_batch` consumes the saved regular Tagesschau inventory JSONL.
It checks the show ID `ff969c14-c5a7-44ab-ab72-14d4c9e427a9`; Tagesschau kompakt
has another ID and is rejected. Historical items are retained as listed in this
show's inventory, including records SRF labels `CLIP`.

```bash
./scripts/uf-python -m src.archive.srf_batch \
  --inventory /path/to/tagesschau.inventory.jsonl \
  --output /path/to/Tegesschau/uf-int16-288p8fps-psnr35 \
  --fps 8 --resolution 288 --target-psnr 35
```

The shortest side becomes 288 pixels, preserving aspect ratio without padding:
16:9 becomes 512×288; 4:3 becomes 384×288. Target PSNR selects a measured preset
from the bundled benchmark; its reference quality is not a per-broadcast
guarantee. Audio uses the archive defaults (mono Opus, 6 kbit/s, 60 ms frames).

The runner processes the inventory newest first, with one persistent INT16 GPU
worker. It selects the highest available source resolution, and extracts video and audio in one
FFmpeg process from one yt-dlp download when the source includes both tracks.
Up to four HLS fragments download concurrently into an owned temporary RAM
directory that is removed on completion or interruption.
Original videos are never stored. Completed output is staged in RAM and
published atomically as the standard UF flat archive (`.bin`, `.opus`,
`.info.json`, `.uf.json`), grouped by year and named by SRF UUID.

Rerun the same command to resume. Existing manifests, profile, source URL and
artifact sizes are checked before contacting the provider; a resumed entry does
not download any source bytes or load the GPU model. Invalid completed archives
are logged as failures and left for review. Normal UF verification remains
available for full artifact hashing and decoding.

`batch-state.json` records the current status and counts; `batch-ledger.jsonl`
records outcomes. Provider metadata and the original inventory row are saved
in each `.info.json`. Geographic restrictions are logged without bypassing
them. Other failures are logged and processing continues; 20 consecutive errors
pause the run to avoid exhausting the inventory during an outage. Rerunning
retries unavailable/failed entries and skips completed ones.

The default free-space reserve is 1 GiB. A conservative output budget is checked
before each download, and actual staged bytes plus the reserve are checked
again before publication. A low-space run exits with code 75 and state
`paused_disk_space`; free space or move the entire output folder, then resume.
The inventory remains separate from the output folder.

For an initial full-program quality check, add
`--ids <one-SRF-UUID> --validate-pilot`. This additionally captures the exact
resized input in `/dev/shm`, closes the encoder worker, decodes the complete UF
stream and reports frame-average YUV420 PSNR (6:1:1), frame counts, artifact
hashes and audio/video duration agreement in `pilot-report.json`. The reference
is removed afterward. This requires enough RAM for the complete raw reference
and must use an entry that has not already been archived. It does not add a
second network download or enable a decode pass for the rest of the batch.
