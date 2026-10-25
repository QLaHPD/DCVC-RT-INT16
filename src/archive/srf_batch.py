"""One-time SRF inventory archiving, using the regular UF encoder and leases.

Run with ``./scripts/uf-python -m src.archive.srf_batch --help``.
No original videos are retained; each muxed source is downloaded only once.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile
from uuid import UUID

from src.archive.media import decoder_thread_budget, dimensions
from src.archive.presets import apply_target_psnr
from src.archive.storage import artifact_path, atomic_json, event, load_bundle, memory_stage_root
from src.archive.streaming import metadata, select_formats
from src.archive.workers import PersistentEncoder
from src.archive.workflow import make_codec, pipeline


SHOW = 'ff969c14-c5a7-44ab-ab72-14d4c9e427a9'
GIB = 1024 ** 3


def classify_error(error):
    """An unavailable item must not consume the controller's failure budget."""
    message = str(error).lower()
    return 'unavailable' if any(word in message for word in (
        'geoblock', 'geo restriction', 'not available in your',
        'only available in switzerland', 'no video formats found',
        'no downloadable video format')) else 'failed'


def read_inventory(path):
    rows, seen = [], set()
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            video_id = str(UUID(row['id']))
            if row.get('show_id') != SHOW or row.get('provider') != 'srf':
                raise ValueError('Inventory contains another show/provider')
            if video_id in seen:
                raise ValueError(f'Duplicate inventory ID: {video_id}')
            row['id'] = video_id
            seen.add(video_id)
            rows.append(row)
    return sorted(rows, key=lambda row: (row['timestamp'], row['id']), reverse=True)


def archive_path(output, row):
    year = datetime.fromtimestamp(row['timestamp'], timezone.utc).strftime('%Y')
    return Path(output) / year / (row['id'] + '.uf.json')


def completed(final, row, config):
    """This must run before contacting SRF or yt-dlp, including on fast resume."""
    if not final.exists():
        return None
    _, data = load_bundle(final, config, verify_hashes=False)
    if data['source'].get('url') != row['url'] or 'video.bin' not in data['artifacts']:
        raise ValueError('Completed archive has a different source or no video')
    if data['source']['media']['audio'] and config['audio'] == 'opus' and 'audio.opus' not in data['artifacts']:
        raise ValueError('Completed archive is missing audio')
    return data


def build_job(row, final, binary, config, reserve):
    info = metadata(row['extractor_url'], binary, None)
    if info.get('id') != row['id']:
        raise ValueError('SRF returned a different video ID')
    # Downscale the highest available resolution, preserving source detail.
    # SRF's HLS renditions include audio, so both tracks share one download.
    video, audio = select_formats(info, max_height=0)
    muxed = video.get('acodec') not in (None, 'none')
    if muxed:
        audio = video
    original = dict(width=int(video['width']), height=int(video['height']),
                    fps=float(video.get('fps') or info.get('fps') or 25),
                    duration=float(info.get('duration') or row['duration_seconds']),
                    audio=audio is not None)
    remote = dict(provider='srf', url=row['url'], binary=binary, cookies=None,
                  media=original, video_format=video['format_id'],
                  audio_format=audio['format_id'] if audio else None,
                  single_pass_audio=muxed, write_thumbnail=False,
                  public_metadata={**info, 'inventory': row,
                                   'uf_source_selection': dict(policy='highest-available-resolution',
                                       video_format=video['format_id'],
                                       audio_format=audio['format_id'] if audio else None,
                                       width=original['width'], height=original['height'])})
    return dict(source=row['id'], relative='Tagesschau/' + row['id'], final=str(final),
                layout='flat', lease_key=row['id'] + '.uf', remote=remote,
                input_threads=decoder_thread_budget(1), prefetch_frames=8,
                resume_check='fast', min_free_bytes=reserve)


def space_budget(duration, bytes_per_second):
    # Reserve a margin over previous output sizes, including a small minimum
    # for metadata and short reports. Publication checks actual bytes again.
    return max(64 * 1024 ** 2, int(max(duration, 1) * bytes_per_second * 2))


class ReferenceSink:
    """Bounded-memory, frame-average PSNR, matching the bundled benchmark."""
    def __init__(self, stream, width, height, frames):
        from src.archive.pareto import PsnrSink
        self.size = width * height * 3 // 2
        self.stream, self.frames = stream, frames
        class SequentialReferences:
            def __len__(_self):
                return self.frames

            def __getitem__(_self, _index):
                reference = self.stream.read(self.size)
                if len(reference) != self.size:
                    raise ValueError('Decoder/reference frame counts differ')
                return reference
        self.metric = PsnrSink(SequentialReferences(), width, height)

    def write(self, raw):
        # HT-L emits a packet containing up to eight frames in one sink write.
        if len(raw) % self.size:
            raise ValueError('Decoder produced a partial frame')
        for offset in range(0, len(raw), self.size):
            self.metric.write(raw[offset:offset + self.size])

    def result(self):
        if self.metric.index != self.frames or self.stream.read(1):
            raise ValueError('Decoder/reference frame counts differ')
        return self.metric.result()


def validate_pilot(final, reference, config, paths, device):
    root, data = load_bundle(final, config)
    codec = make_codec(config, paths, device)
    with Path(reference).open('rb') as stream:
        video = data['video']
        sink = ReferenceSink(stream, video['width'], video['height'], video['frames'])
        decoded = codec.decode(artifact_path(root, data, 'video.bin'), video, sink=sink,
                               progress=lambda **values: event('pilot_decode_progress', **values))
        quality = sink.result()
    audio_duration = None
    if 'audio.opus' in data['artifacts']:
        import subprocess
        result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                                 '-of', 'json', str(artifact_path(root, data, 'audio.opus'))],
                                capture_output=True, text=True, check=True)
        audio_duration = float(json.loads(result.stdout)['format']['duration'])
        if abs(audio_duration - video['frames'] / video['fps']) > 1:
            raise ValueError('Pilot video/audio durations differ by more than one second')
    report = dict(archive=str(final), video=video, decoded=decoded, **quality,
                  audio_duration=audio_duration,
                  archive_bytes=sum(item['size'] for item in data['artifacts'].values()),
                  quality_basis=f"frame-average YUV420 PSNR, weighted 6:1:1; exact resized {video['fps']:g}fps input")
    atomic_json(final.parent.parent / 'pilot-report.json', report)
    event('pilot_validated', **report)
    return report


def codec_options(args):
    from main import build_parser
    options = build_parser().parse_args(['encode', '--runtime', 'int16', '--target-psnr',
                                         str(args.target_psnr), '--fps', str(args.fps),
                                         '--resolution', str(args.resolution)])
    apply_target_psnr(options)
    return options


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # A single sequential scheduler bounds GPU memory and makes space checks
    # effective immediately; per-video leases remain compatible with UF tools.
    with (output / '.batch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_locked(args, output)


def run_locked(args, output):
    rows = read_inventory(args.inventory)
    if args.ids:
        ids = {str(UUID(value)) for value in args.ids}
        rows = [row for row in rows if row['id'] in ids]
        if len(rows) != len(ids):
            raise ValueError('Requested video not found in the inventory')
    if args.validate_pilot and (len(rows) != 1 or archive_path(output, rows[0]).exists()):
        raise ValueError('--validate-pilot requires one --ids video that is not already archived')
    binary = shutil.which(args.ytdlp_bin)
    if not binary:
        raise ValueError(f'yt-dlp not found: {args.ytdlp_bin}')
    options = codec_options(args)
    config, paths = pipeline(options)
    settings = dict(pipeline=config, preset=options.target_psnr_selection,
                    source_selection='highest-available-resolution')
    settings_path = output / 'batch-settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError('Output folder already uses different encoding settings')
    atomic_json(settings_path, settings)
    reserve = int(args.min_free_gib * GIB)
    counts = Counter()
    state = dict(status='running', inventory=str(Path(args.inventory).resolve()), total=len(rows),
                 settings=settings, started_at=datetime.now(timezone.utc).isoformat())
    rate = 8192.0
    worker, reference_dir = None, None
    consecutive_failures = 0

    def save(status, **extra):
        state.update(status=status, counts=dict(counts), free_bytes=shutil.disk_usage(output).free,
                     updated_at=datetime.now(timezone.utc).isoformat(), **extra)
        atomic_json(output / 'batch-state.json', state)

    def record(row, status, **extra):
        counts[status] += 1
        with (output / 'batch-ledger.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(),
                                         id=row['id'], status=status, **extra)) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        event('srf_job_settled', video_id=row['id'], status=status, counts=dict(counts), **extra)
        save('running')

    save('running')
    try:
        for row in rows:
            if args.limit and sum(counts.values()) >= args.limit:
                break
            final = archive_path(output, row)
            save('running', current_id=row['id'], current_title=row['title'])
            try:
                previous = completed(final, row, config)
                if previous is not None:
                    record(row, 'resumed', archive=str(final))
                    continue
                if shutil.disk_usage(output).free < reserve + space_budget(row['duration_seconds'], rate):
                    save('paused_disk_space', next_id=row['id'])
                    event('srf_paused_disk_space', free_bytes=shutil.disk_usage(output).free, reserve_bytes=reserve)
                    return 75
                event('srf_resolving', video_id=row['id'], title=row['title'])
                job = build_job(row, final, binary, config, reserve)
                if args.validate_pilot:
                    media = job['remote']['media']
                    width, height = dimensions(media['width'], media['height'], config['resolution'])
                    needed = width * height * 3 // 2 * (int(media['duration'] * args.fps) + 2)
                    if shutil.disk_usage(memory_stage_root()).free < needed + GIB:
                        raise ValueError('Insufficient RAM space for the full pilot reference')
                    reference_dir = tempfile.TemporaryDirectory(prefix='uf-srf-pilot-', dir=memory_stage_root())
                    job['reference_output'] = str(Path(reference_dir.name) / 'reference.yuv')
                if worker is None:
                    worker = PersistentEncoder(multiprocessing.get_context('spawn'), config, paths,
                                               args.cuda_idx, 'srf-batch', None)
                worker.assign(job)
                while worker.process.is_alive() and not worker.connection.poll(.5):
                    pass
                status = worker.receive()
                if status == 'failed':
                    worker.close()
                    worker = None
                if status == 'encoded':
                    data = completed(final, row, config)
                    size = sum(item['size'] for item in data['artifacts'].values())
                    duration = data['video']['frames'] / data['video']['fps']
                    rate = max(rate, size / max(duration, 1))
                    record(row, status, archive=str(final), bytes=size, duration=duration)
                    consecutive_failures = 0
                    if args.validate_pilot:
                        worker.close()
                        worker = None
                        validate_pilot(final, job['reference_output'], config, paths, args.cuda_idx)
                else:
                    record(row, status, archive=str(final))
                    consecutive_failures += status == 'failed'
                    if args.validate_pilot:
                        save('pilot_failed')
                        return 1
            except Exception as exc:
                unavailable = classify_error(exc) == 'unavailable'
                record(row, classify_error(exc), error=f'{type(exc).__name__}: {exc}')
                consecutive_failures = 0 if unavailable else consecutive_failures + 1
                if args.validate_pilot:
                    save('pilot_failed', error=str(exc))
                    return 1
            if consecutive_failures >= args.max_consecutive_failures:
                save('paused_errors', message='Consecutive failures; check network/worker logs before resuming')
                return 1
        save('finished', current_id=None, current_title=None)
        return int(bool(counts['failed']))
    except KeyboardInterrupt:
        save('interrupted')
        return 130
    finally:
        if worker is not None:
            worker.close(abort=worker.job is not None)
        if reference_dir is not None:
            reference_dir.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', required=True, help='Saved SRF inventory JSONL; regular Tagesschau show only')
    parser.add_argument('--output', required=True)
    parser.add_argument('--ids', nargs='+', help='Optional inventory IDs, for a pilot/subset')
    parser.add_argument('--limit', type=int, help='Stop after this many entries, including resumed/failed entries')
    parser.add_argument('--fps', type=float, default=8)
    parser.add_argument('--resolution', type=int, default=288, help='Shortest side; preserve aspect ratio, no bars')
    parser.add_argument('--target-psnr', type=float, default=35, help='Bundled benchmark target, not a per-video guarantee')
    parser.add_argument('--cuda-idx', type=int, default=0)
    parser.add_argument('--ytdlp-bin', default='yt-dlp')
    parser.add_argument('--min-free-gib', type=float, default=1)
    parser.add_argument('--max-consecutive-failures', type=int, default=20)
    parser.add_argument('--validate-pilot', action='store_true', help='Capture a full RAM reference and decode/measure PSNR')
    args = parser.parse_args()
    if (args.fps <= 0 or args.resolution < 2 or args.resolution % 2 or args.min_free_gib < 0
            or args.max_consecutive_failures < 1 or (args.limit is not None and args.limit < 1)):
        parser.error('Require positive fps, positive even short edge, nonnegative reserve and positive limits')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
