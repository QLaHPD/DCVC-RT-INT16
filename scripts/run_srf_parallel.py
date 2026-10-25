#!/usr/bin/env python3
"""Resume an exported Tagesschau batch with one persistent encoder per GPU."""
import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import socket
import sys
import threading
import time

MODEL_I = 'cvpr2026_image.pth.tar.uf-int16-q9-w13-acc64-rna-sat-v1.b3b900de23f30e4f.int16.pt'
MODEL_P = 'cvpr2026_video_htl.pth.tar.uf-int16-q9-w13-acc64-rna-sat-v1.934bde4a12fc0b6b.int16.pt'


def now():
    return datetime.now(timezone.utc).isoformat()


def gpu_ids(value):
    try:
        devices = [int(part) for part in value.split(',')]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Use comma-separated CUDA indices, e.g. 0,1,2,3,4') from exc
    if not devices or min(devices) < 0 or len(devices) != len(set(devices)):
        raise argparse.ArgumentTypeError('CUDA indices must be nonnegative and unique')
    return devices


def classify_error(error):
    from src.archive.srf_batch import classify_error as classify
    return classify(error)


def prepare_options(repo, config, prepared_i, prepared_p, procs):
    from main import build_parser
    from src.archive.profiles import validate_options
    from src.archive.workflow import pipeline
    # Recover concrete saved parameters, rather than reselecting a benchmark.
    options = build_parser().parse_args([
        'encode', '--runtime', config['runtime'], '--model_structure', config['variant'],
        '--fps', str(config['fps']), '--resolution', str(config['resolution']),
        '--qi', str(config['qp_i']), '--qp', str(config['qp_p']),
        '--reset_interval', str(config['reset_interval']),
        '--intra_period', str(config['intra_period']),
        '--skip_thres', str(config.get('skip_threshold', 0)),
        '--audio', config['audio'], '--opus_channels', config['opus_channels'],
        '--opus_bitrate', config['opus_bitrate'],
        '--opus_frame_ms', str(config.get('opus_frame_ms', 20)),
        '--opus_complexity', str(config.get('opus_complexity', 10)),
        '--opus_vbr', config.get('opus_vbr', 'on'), '--procs', str(procs),
        '--prepared_i', str(prepared_i or repo / 'checkpoints' / MODEL_I),
        '--prepared_p', str(prepared_p or repo / 'checkpoints' / MODEL_P)])
    validate_options(options)
    actual, paths = pipeline(options)
    if actual != config:
        raise ValueError('Local models/options do not reproduce the exported pipeline; do not edit batch-settings.json')
    return paths


def resolve_job(row, final, binary, config, reserve, stop, procs):
    from src.archive.media import decoder_thread_budget
    from src.archive.streaming import metadata, select_formats
    info = metadata(row['extractor_url'], binary, None, cancel_event=stop)
    if info.get('id') != row['id']:
        raise ValueError('SRF returned a different video ID')
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
                input_threads=decoder_thread_budget(procs), prefetch_frames=8,
                resume_check='fast', min_free_bytes=reserve)


class Scheduler:
    """Bound metadata requests and persistent encoders to one slot per GPU."""
    def __init__(self, rows, devices, resolve, make_worker, completed, settle,
                 space_ok, state, stop=None, error_limit=20):
        self.rows = deque(rows)
        self.slots = [dict(device=device, worker=None, future=None, row=None) for device in devices]
        self.resolve, self.make_worker, self.completed = resolve, make_worker, completed
        self.settle, self.space_ok, self.state = settle, space_ok, state
        self.stop = stop or threading.Event()
        self.error_limit = error_limit
        self.failures = 0

    def result(self, row, status, **extra):
        self.settle(row, status, **extra)
        if status in ('encoded', 'resumed'):
            self.failures = 0
        elif status == 'failed':
            self.failures += 1
        if self.failures >= self.error_limit:
            raise RuntimeError(f'{self.error_limit} consecutive failures; inspect the log before restarting')

    def run(self):
        pool = ThreadPoolExecutor(max_workers=len(self.slots), thread_name_prefix='srf-metadata')
        paused_space = False
        try:
            while self.rows or any(slot['row'] is not None for slot in self.slots):
                if self.stop.is_set():
                    raise KeyboardInterrupt
                for slot in self.slots:
                    row, worker = slot['row'], slot['worker']
                    if row is not None and worker is not None and worker.job is not None:
                        if worker.connection.poll() or not worker.process.is_alive():
                            status = worker.receive()
                            if status == 'failed':
                                worker.close()
                                slot['worker'] = None
                            slot['row'] = None
                            self.result(row, status, gpu=slot['device'])
                        continue
                    future = slot['future']
                    if future is not None:
                        if not future.done():
                            continue
                        slot['future'] = None
                        try:
                            job = future.result()
                            already = self.completed(row)
                            if not already:
                                if slot['worker'] is None:
                                    slot['worker'] = self.make_worker(slot['device'])
                                slot['worker'].assign(job)
                        except Exception as exc:
                            if slot['worker'] is not None:
                                slot['worker'].close(abort=True)
                                slot['worker'] = None
                            slot['row'] = None
                            self.result(row, classify_error(exc), gpu=slot['device'], error=f'{type(exc).__name__}: {exc}')
                        else:
                            if already:
                                slot['row'] = None
                                self.result(row, 'resumed', gpu=slot['device'])
                            else:
                                self.state('running', self.slots)
                        continue
                    if row is None and self.rows and not paused_space:
                        candidate = self.rows[0]
                        if not self.space_ok(candidate):
                            paused_space = True
                            self.state('draining_low_space', self.slots)
                            continue
                        slot['row'] = self.rows.popleft()
                        slot['future'] = pool.submit(self.resolve, slot['row'])
                        self.state('running', self.slots)
                if paused_space and not any(slot['row'] is not None for slot in self.slots):
                    self.state('paused_disk_space', self.slots)
                    return 75
                self.stop.wait(.1)
            self.state('finished', self.slots)
            return 0
        except KeyboardInterrupt:
            self.state('interrupted', self.slots)
            raise
        except Exception:
            self.state('paused_errors', self.slots)
            raise
        finally:
            self.stop.set()
            for slot in self.slots:
                if slot['future'] is not None:
                    slot['future'].cancel()
                worker = slot['worker']
                if worker is not None:
                    worker.close(abort=worker.job is not None)
            pool.shutdown(wait=True, cancel_futures=True)


def run(args):
    repo, root = Path(args.repo).resolve(), Path(args.root).resolve()
    sys.path.insert(0, str(repo))
    from src.archive.srf_batch import archive_path, completed, read_inventory, space_budget
    from src.archive.storage import atomic_json, event
    inventory = root / 'inventory-2026-10-18/tagesschau.inventory.jsonl'
    output = root / 'uf-int16-288p8fps-psnr35'
    settings = json.loads((output / 'batch-settings.json').read_text())
    config = settings['pipeline']
    if (config['runtime'], config['variant'], config['fps'], config['resolution'],
        config['qp_i'], config['qp_p'], config['reset_interval'], config['intra_period'],
        config['audio'], config['opus_channels'], config['opus_bitrate'],
        config.get('opus_frame_ms'), config.get('skip_threshold', 0), config['max_frames']) != (
            'int16', 'htl', 8, 288, 15, 39, 3, -1, 'opus', 'mono', '6k', 60, 0, None):
        raise ValueError('Exported settings differ from the approved Tagesschau profile')
    if settings.get('source_selection') != 'highest-available-resolution':
        raise ValueError('The exported archive does not use highest-available source selection')
    rows = read_inventory(inventory)
    counts = Counter()
    pending, invalid = [], []
    print(json.dumps(dict(event='preflight_started', inventory=len(rows))), flush=True)
    for row in rows:
        try:
            if completed(archive_path(output, row), row, config) is not None:
                counts['resumed'] += 1
            else:
                pending.append(row)
        except Exception as exc:
            invalid.append(dict(id=row['id'], error=f'{type(exc).__name__}: {exc}'))
    print(json.dumps(dict(event='preflight_complete', total=len(rows), counts=dict(counts),
                          pending=len(pending), invalid=invalid), ensure_ascii=False), flush=True)
    if invalid:
        raise ValueError('Existing archives failed validation; preserved for review, nothing downloaded')
    if args.check_only:
        return 0
    paths = prepare_options(repo, config, args.prepared_i, args.prepared_p, len(args.gpus))
    binary = shutil.which(args.ytdlp_bin)
    if not binary:
        raise ValueError('yt-dlp is missing from the selected environment; specify --ytdlp-bin')
    import torch
    if not torch.cuda.is_available() or max(args.gpus) >= torch.cuda.device_count():
        raise ValueError('One or more requested CUDA devices are not visible')
    from src.archive.workers import PersistentEncoder
    reserve = int(args.min_free_gib * 1024**3)
    started = now()
    stop = threading.Event()
    state_path = output / 'parallel-state.json'
    ledger = output / 'parallel-ledger.jsonl'
    instance = 'srf-a4000-' + socket.gethostname() + '-' + str(os.getpid())
    rates = [8192.0]

    def state(status, slots):
        atomic_json(state_path, dict(status=status, started_at=started, updated_at=now(),
            total=len(rows), counts=dict(counts), gpus=args.gpus,
            active=[dict(gpu=s['device'], id=s['row']['id']) for s in slots if s['row']],
            free_bytes=shutil.disk_usage(output).free, inventory=str(inventory), pipeline=config))

    def settle(row, status, **extra):
        if status in ('encoded', 'resumed'):
            data = completed(archive_path(output, row), row, config)
            if data is None:
                raise ValueError('Worker reported success without a committed archive')
            size = sum(record['size'] for record in data['artifacts'].values())
            duration = data['video']['frames'] / data['video']['fps']
            rates[0] = max(rates[0], size / max(duration, 1))
            extra.update(bytes=size, duration=duration)
        counts[status] += 1
        with ledger.open('a') as f:
            f.write(json.dumps(dict(time=now(), id=row['id'], status=status, **extra)) + '\n')
            f.flush()
            os.fsync(f.fileno())
        event('srf_parallel_settled', video_id=row['id'], status=status, counts=dict(counts), **extra)
        state('running', scheduler.slots)

    def space_ok(row):
        # Reserve a conservative budget for every in-flight and proposed job.
        durations = [row['duration_seconds']] + [s['row']['duration_seconds'] for s in scheduler.slots if s['row']]
        needed = reserve + sum(space_budget(duration, rates[0]) for duration in durations)
        return shutil.disk_usage(output).free >= needed

    scheduler = Scheduler(pending, args.gpus,
        resolve=lambda row: resolve_job(row, archive_path(output, row), binary, config, reserve, stop, len(args.gpus)),
        make_worker=lambda device: PersistentEncoder(multiprocessing.get_context('spawn'), config, paths, device, instance, None),
        completed=lambda row: completed(archive_path(output, row), row, config) is not None,
        settle=settle, space_ok=space_ok, state=state, stop=stop,
        error_limit=args.max_consecutive_failures)
    state('running', scheduler.slots)
    status = scheduler.run()
    return status or int(bool(counts['failed'] or counts['busy']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, help='Existing local DCVC-UF-INT16 checkout')
    parser.add_argument('--root', required=True, help='Extracted Tegesschau directory')
    parser.add_argument('--gpus', type=gpu_ids, default=gpu_ids('0,1,2,3,4'))
    parser.add_argument('--prepared-i', type=Path)
    parser.add_argument('--prepared-p', type=Path)
    parser.add_argument('--ytdlp-bin', default='yt-dlp')
    parser.add_argument('--min-free-gib', type=float, default=5)
    parser.add_argument('--max-consecutive-failures', type=int, default=20)
    parser.add_argument('--check-only', action='store_true', help='Offline archive preflight; no models, GPU loads, or downloads')
    args = parser.parse_args()
    if args.min_free_gib < 0 or args.max_consecutive_failures < 1:
        parser.error('Reserve must be nonnegative and failure limit positive')
    output = Path(args.root).resolve() / 'uf-int16-288p8fps-psnr35'
    # This is the same controller lock as the original single-GPU runner.
    with (output / '.batch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            return run(args)
        except KeyboardInterrupt:
            return 130


if __name__ == '__main__':
    raise SystemExit(main())
