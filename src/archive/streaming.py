"""YouTube inputs streamed through yt-dlp into the normal UF archive encoder."""
from contextlib import contextmanager
import json
import multiprocessing
import os
from pathlib import Path
from src.archive.dashboard import event_queue
import re
import shutil
import subprocess
import tempfile

from src.archive.storage import event


@contextmanager
def cookie_copy(source):
    if not source:
        yield None
        return
    fd, name = tempfile.mkstemp(prefix='uf-ytdlp-cookies-')
    try:
        os.fchmod(fd, 0o660)
        with os.fdopen(fd, 'wb') as target, open(source, 'rb') as original:
            shutil.copyfileobj(original, target)
        yield name
    finally:
        Path(name).unlink(missing_ok=True)


def command(binary, cookies):
    deno = shutil.which('deno') or str(Path.home() / '.deno/bin/deno')
    runtime = ['--js-runtimes', 'deno:' + deno] if Path(deno).is_file() else []
    return [binary, *runtime, '--ignore-config', '--remote-components', 'ejs:github', '--no-progress', '--retries', '5',
            '--fragment-retries', '5', *(['--cookies', cookies] if cookies else [])]


def metadata(url, binary, cookies, flat=False):
    with cookie_copy(cookies) as copy:
        cmd = command(binary, copy) + ['--dump-single-json', '--skip-download']
        cmd += ['--flat-playlist', '--ignore-errors'] if flat else ['--no-playlist']
        result = subprocess.run(cmd + ['--', url], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr[-2500:])
        return json.loads(result.stdout)


def safe_id(value, channel=False):
    pattern = r'UC[A-Za-z0-9_-]{22}' if channel else r'[A-Za-z0-9_-]{11}'
    if not isinstance(value, str) or not re.fullmatch(pattern, value):
        raise ValueError(f'Invalid YouTube {"channel" if channel else "video"} ID: {value!r}')
    return value


def select_formats(info):
    formats = [f for f in info.get('formats', []) if f.get('url') and f.get('vcodec') not in (None, 'none')
               and f.get('height') and not f.get('has_drm')]
    if not formats:
        raise ValueError('No downloadable video format')
    video = max(formats, key=lambda f: (f.get('height') or 0, f.get('width') or 0, f.get('tbr') or 0))
    audio = [f for f in info.get('formats', []) if f.get('acodec') not in (None, 'none')
             and f.get('url') and not f.get('has_drm')]
    # yt-dlp marks original tracks in format_note / language_preference.
    originals = [f for f in audio if 'original' in str(f.get('format_note', '')).lower()
                 or (f.get('language_preference') or 0) >= 10]
    if originals:
        audio = originals
    elif len({f.get('language') for f in audio if f.get('language')}) > 1:
        raise ValueError('Multiple audio languages without an identified original track; refusing an automatic dub')
    track = max(audio, key=lambda f: (f.get('language_preference') or 0, f.get('abr') or f.get('tbr') or 0)) if audio else None
    return video, track


@contextmanager
def download_pipe(remote, format_id):
    with cookie_copy(remote.get('cookies')) as copy, tempfile.TemporaryFile() as errors:
        cmd = command(remote['binary'], copy) + ['--no-playlist', '-f', str(format_id), '-o', '-', '--', remote['url']]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errors)
        normal = False
        try:
            yield process.stdout
            # FFmpeg may stop audio at the encoded video duration. Drain the
            # remaining source bytes so a successful download does not get SIGPIPE.
            while process.stdout.read(1024 * 1024):
                pass
            normal = True
        finally:
            process.stdout.close()
            try:
                code = process.wait(timeout=15 if normal else 1)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    code = process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    code = process.wait()
            if normal and code:
                errors.seek(0)
                raise RuntimeError('yt-dlp stream failed: ' + errors.read().decode(errors='replace')[-2500:])


def run_stream(args):
    from src.archive.workflow import pipeline, _job_entry
    from src.archive.media import decoder_thread_budget
    if args.procs != 1:
        raise ValueError('UF YouTube streaming currently requires --procs 1')
    if args.max_frames:
        raise ValueError('UF streaming currently requires full videos; omit --max_frames')
    binary = shutil.which(args.ytdlp_bin)
    if not binary:
        raise ValueError('yt-dlp not found; pass --ytdlp_bin /path/to/yt-dlp')
    args._cleanup_sources = set()  # Streamed jobs have no local originals.
    config, paths = pipeline(args)
    context = multiprocessing.get_context('spawn')
    counts = {'encoded': 0, 'resumed': 0, 'busy': 0, 'failed': 0}
    seen = set()
    for url in args.source_urls:
        listing = metadata(url, binary, args.cookies, flat=True)
        entries = listing.get('entries') if listing.get('_type') in ('playlist', 'multi_video') else [listing]
        if entries is None:
            entries = [listing]
        # Channel handles can resolve to a list of tabs.
        pending = list(entries)
        while pending:
            entry = pending.pop(0)
            if not entry:
                continue
            if entry.get('_type') in ('playlist', 'multi_video'):
                sub = entry if entry.get('entries') is not None else metadata(entry['url'], binary, args.cookies, flat=True)
                pending.extend(sub.get('entries') or [])
                continue
            try:
                video_id = safe_id(entry.get('id'))
                if video_id in seen:
                    continue
                seen.add(video_id)
                video_url = 'https://www.youtube.com/watch?v=' + video_id
                event('stream_resolving', video_id=video_id)
                info = metadata(video_url, binary, args.cookies)
                if info.get('is_live') or info.get('live_status') == 'is_upcoming':
                    event('stream_skipped_live', video_id=video_id)
                    continue
                channel = safe_id(info.get('channel_id'), channel=True)
                video, audio = select_formats(info)
                original = dict(width=int(video['width']), height=int(video['height']),
                                fps=float(video.get('fps') or info.get('fps') or 30),
                                duration=float(info.get('duration') or 0), audio=audio is not None)
                public = {k: info.get(k) for k in ('id', 'channel_id', 'channel', 'title', 'description', 'upload_date', 'duration', 'webpage_url')}
                remote = dict(url=video_url, media=original, binary=binary, cookies=args.cookies,
                              video_format=video['format_id'], audio_format=audio['format_id'] if audio else None,
                              public_metadata=public)
                legacy = Path(args.output_root).resolve() / channel / (video_id + '.uf')
                final = legacy.with_name(video_id + '.uf.json')
                job = dict(source=video_id, relative=channel+'/'+video_id, final=str(final), remote=remote,
                           layout='flat', legacy=str(legacy), lease_key=legacy.name,
                           input_threads=args.input_threads or decoder_thread_budget(1),
                           prefetch_frames=args.prefetch_frames if args.prefetch_frames is not None else 8)
                event('stream_selected', video_id=video_id, channel_id=channel, video_format=remote['video_format'],
                      audio_format=remote['audio_format'], audio_language=audio.get('language') if audio else None,
                      audio_note=audio.get('format_note') if audio else None, archive=str(final))
                parent, child = context.Pipe(duplex=False)
                worker = context.Process(target=_job_entry, args=(child, job, config, paths,
                     'cpu' if args.device == 'cpu' else args.cuda_idx[0], args.shared_instance),
                    kwargs={'event_queue': event_queue()})
                try:
                    worker.start();child.close()
                    while worker.is_alive() and not parent.poll(1):
                        pass
                    outcome = parent.recv() if parent.poll() else 'failed'
                    worker.join()
                finally:
                    if worker.is_alive():worker.terminate();worker.join()
                    parent.close();child.close()
                counts[outcome if outcome in counts else 'failed'] += 1
            except Exception as exc:
                counts['failed'] += 1
                event('stream_failed', video_id=entry.get('id'), error=str(exc))
    event('stream_finished', **counts)
    from src.archive.lifecycle import finish
    from src.archive.dashboard import command_queue
    # Streaming never has local originals to delete; show the completed channel.
    finish(args, command_queue(), [str(Path(args.output_root).resolve())] if counts['failed'] else ())
    return int(counts['failed'] > 0)
