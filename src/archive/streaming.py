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


def metadata(url, binary, cookies, flat=False, cancel_event=None):
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError('Streaming metadata cancelled')
    with cookie_copy(cookies) as copy:
        cmd = command(binary, copy) + ['--dump-single-json', '--skip-download']
        cmd += ['--flat-playlist', '--ignore-errors'] if flat else ['--no-playlist']
        if cancel_event is None:
            result = subprocess.run(cmd + ['--', url], capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stderr[-2500:])
            return json.loads(result.stdout)
        process = subprocess.Popen(cmd + ['--',url], stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
        try:
            while True:
                if cancel_event.is_set():
                    raise InterruptedError('Streaming metadata cancelled')
                try:
                    output,errors = process.communicate(timeout=.2)
                    break
                except subprocess.TimeoutExpired:
                    pass
            if process.returncode: raise RuntimeError(errors[-2500:])
            return json.loads(output)
        finally:
            if process.poll() is None:
                import signal
                try: os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                process.wait()
            process.stdout.close(); process.stderr.close()


def safe_id(value, channel=False):
    pattern = r'UC[A-Za-z0-9_-]{22}' if channel else r'[A-Za-z0-9_-]{11}'
    if not isinstance(value, str) or not re.fullmatch(pattern, value):
        raise ValueError(f'Invalid YouTube {"channel" if channel else "video"} ID: {value!r}')
    return value


def select_formats(info, max_height=0):
    formats = [f for f in info.get('formats', []) if f.get('url') and f.get('vcodec') not in (None, 'none')
               and f.get('height') and not f.get('has_drm')]
    if not formats:
        raise ValueError('No downloadable video format')
    if max_height:
        bounded = [f for f in formats if f['height'] <= max_height]
        formats = bounded or [f for f in formats if f['height'] == min(v['height'] for v in formats)]
    best_geometry = max((f.get('height') or 0, f.get('width') or 0) for f in formats)
    best_formats = [f for f in formats
                    if ((f.get('height') or 0, f.get('width') or 0) == best_geometry)]
    separate_video = [f for f in best_formats if f.get('acodec') in (None, 'none')]
    if separate_video:
        best_formats = separate_video
    video = max(best_formats, key=lambda f: f.get('tbr') or 0)
    audio = [f for f in info.get('formats', []) if f.get('acodec') not in (None, 'none')
             and f.get('url') and not f.get('has_drm')]
    separate_audio = [f for f in audio if f.get('vcodec') in (None, 'none')]
    if separate_audio:
        audio = separate_audio
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
def download_pipe(remote, format_id, prepared=None):
    if prepared is not None:
        from src.archive.stream_pipeline import prepared_pipe
        with prepared_pipe(prepared) as stream:
            yield stream
        return
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



def upload_timestamp(info):
    """Prefer provider Unix time; date-only metadata means midnight UTC."""
    from datetime import datetime, timezone
    import math
    for key in ('timestamp','release_timestamp'):
        value = info.get(key)
        if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value >= 0:
            return int(value)
    date = info.get('upload_date')
    if isinstance(date,str) and re.fullmatch(r'\d{8}',date):
        return int(datetime.strptime(date,'%Y%m%d').replace(tzinfo=timezone.utc).timestamp())
    raise ValueError('Remote video has no publication timestamp or upload date')


def existing_stream_archives(channel_out, completed=None, config=None, resume_check='fast'):
    """One metadata snapshot per channel, preserving prior naming on resume."""
    archives = {}
    for path in Path(channel_out).glob('*.uf.json'):
        data = json.loads(path.read_text())
        url = data.get('source',{}).get('url')
        if data.get('format') != 'dcvc-uf-archive' or not url: continue
        if url in archives:
            raise ValueError(f'Multiple archives for {url}: {archives[url]} and {path}')
        archives[url] = path
        if completed is not None and config is not None and not path.is_symlink():
            from src.archive.storage import has_completion_record, artifact_path, sha256
            artifacts = data.get('artifacts', {})
            if data.get('pipeline') != config or not has_completion_record(data) or 'video.bin' not in artifacts:
                continue
            if config.get('audio') == 'opus' and data.get('source', {}).get('media', {}).get('audio') and 'audio.opus' not in artifacts:
                continue
            try:
                if all(not (artifact := artifact_path(path.parent, data, name)).is_symlink()
                       and artifact.is_file() and artifact.stat().st_size == record['size']
                       and (resume_check != 'full' or sha256(artifact) == record['sha256'])
                       for name, record in artifacts.items()):
                    completed.add(url)
            except (OSError, ValueError, KeyError, TypeError):
                pass
    return archives

def stream_jobs(args, binary, counts, cancel_event=None, config=None):
    if hasattr(args, '_stream_catalog'):
        yield from catalog_jobs(args, binary, counts, cancel_event)
        return
    from src.archive.media import decoder_thread_budget
    seen = set()
    archives, completed = {}, {}
    scheduled = 0
    def snapshot(channel):
        if channel not in archives:
            completed[channel] = set()
            archives[channel] = existing_stream_archives(Path(args.output_root).resolve()/channel,
                                                         completed[channel], config,
                                                         getattr(args, 'resume_check', 'fast'))
        return completed[channel]
    for url in args.source_urls:
        if cancel_event is not None and cancel_event.is_set(): return
        listing = metadata(url, binary, args.cookies, flat=True, cancel_event=cancel_event)
        entries = listing.get('entries') if listing.get('_type') in ('playlist', 'multi_video') else [listing]
        if entries is None:
            entries = [listing]
        # Channel handles can resolve to a list of tabs.
        pending = [(entry, listing.get('channel_id')) for entry in entries]
        while pending:
            if cancel_event is not None and cancel_event.is_set(): return
            entry, channel_hint = pending.pop(0)
            if not entry:
                continue
            if entry.get('_type') in ('playlist', 'multi_video'):
                sub = entry if entry.get('entries') is not None else metadata(entry['url'], binary, args.cookies, flat=True, cancel_event=cancel_event)
                pending.extend((item, sub.get('channel_id') or channel_hint) for item in (sub.get('entries') or []))
                continue
            try:
                youtube = 'twitch.tv' not in url.lower()
                video_id = safe_id(entry.get('id')) if youtube else safe_component(entry.get('id'))
                if video_id in seen:
                    continue
                seen.add(video_id)
                video_url = 'https://www.youtube.com/watch?v=' + video_id if youtube else (entry.get('webpage_url') or entry.get('url') or 'https://www.twitch.tv/videos/'+video_id.lstrip('v'))
                hint = entry.get('channel_id') or channel_hint
                if hint:
                    hint = safe_id(hint, channel=True) if youtube else safe_component(hint)
                    if video_url in snapshot(hint):
                        counts['resumed'] = counts.get('resumed', 0) + 1
                        continue
                if getattr(args,'max_videos',None) and scheduled >= args.max_videos:
                    return
                scheduled += 1
                event('run_discovered', total=scheduled+len(pending))
                event('stream_resolving', video_id=video_id)
                info = metadata(video_url, binary, args.cookies, cancel_event=cancel_event)
                if info.get('is_live') or info.get('live_status') == 'is_upcoming':
                    event('stream_skipped_live', video_id=video_id)
                    continue
                channel = safe_id(info.get('channel_id'), channel=True) if youtube else safe_component(info.get('channel_id') or info.get('uploader_id'))
                video, audio = select_formats(info, getattr(args,"source_max_height",480))
                original = dict(width=int(video['width']), height=int(video['height']),
                                fps=float(video.get('fps') or info.get('fps') or 30),
                                duration=float(info.get('duration') or 0), audio=audio is not None)
                # Keep the complete yt-dlp document, matching RT's info.json.
                # The legacy job key is retained for compatibility with workers.
                public = info
                remote = dict(url=video_url, media=original, binary=binary, cookies=args.cookies,
                              video_format=video['format_id'], audio_format=audio['format_id'] if audio else None,
                              public_metadata=public, thumbnail=info.get('thumbnail'),thumbnail_headers=info.get('http_headers'), write_thumbnail=getattr(args,'write_thumbnail',True),
                              thumbnail_codec=getattr(args,'thumbnail_codec','keep'),thumbnail_qp=getattr(args,'thumbnail_qp',45))
                legacy = Path(args.output_root).resolve() / channel / (video_id + '.uf')
                if video_url in snapshot(channel):
                    counts['resumed'] = counts.get('resumed', 0) + 1
                    scheduled -= 1
                    continue
                final = archives[channel].get(video_url)
                if final is None:
                    final = legacy.with_name(f'{video_id}_{upload_timestamp(info)}.uf.json')
                job = dict(source=video_id, relative=channel+'/'+video_id, final=str(final), remote=remote,
                           resume_check=getattr(args, 'resume_check', 'fast'),
                           hwaccel=getattr(args,'ff_hwaccel','none'), layout='flat', filename_style='rt', legacy=str(legacy), lease_key=legacy.name,lease_seconds=getattr(args,'shared_lease_seconds',900),
                           input_threads=args.input_threads or decoder_thread_budget(1),
                           prefetch_frames=args.prefetch_frames if args.prefetch_frames is not None else 8)
                event('stream_selected', video_id=video_id, channel_id=channel, video_format=remote['video_format'],
                      audio_format=remote['audio_format'], audio_language=audio.get('language') if audio else None,
                      audio_note=audio.get('format_note') if audio else None, archive=str(final))
                yield job
            except Exception as exc:
                counts['failed'] += 1
                event('stream_failed', video_id=entry.get('id'), error=str(exc))
    event('run_discovered', total=scheduled)


def stream_catalog(args, binary):
    """Resolve channel identities before any encoder or downloader starts."""
    entries, failures, seen = [], [], set()
    for url in args.source_urls:
        listing = metadata(url, binary, args.cookies, flat=True)
        pending = [(listing, listing.get('channel_id'))]
        youtube = 'twitch.tv' not in url.lower()
        while pending:
            entry, hint = pending.pop()
            if not entry:
                continue
            if entry.get('_type') in ('playlist', 'multi_video'):
                sub = entry if entry.get('entries') is not None else metadata(
                    entry['url'], binary, args.cookies, flat=True)
                hint = sub.get('channel_id') or hint
                pending.extend((item, hint) for item in reversed(sub.get('entries') or []))
                continue
            try:
                video_id = safe_id(entry.get('id')) if youtube else safe_component(entry.get('id'))
                video_url = ('https://www.youtube.com/watch?v=' + video_id if youtube else
                             entry.get('webpage_url') or entry.get('url') or
                             'https://www.twitch.tv/videos/' + video_id.lstrip('v'))
                if video_url in seen:
                    continue
                seen.add(video_url)
                channel = entry.get('channel_id') or hint
                if not channel:
                    # Identity-only extraction; no media bytes are downloaded.
                    entry = metadata(video_url, binary, args.cookies)
                    channel = entry.get('channel_id') or entry.get('uploader_id')
                channel = safe_id(channel, channel=True) if youtube else safe_component(channel)
                # Keep only identity data in the catalog. Full format lists are
                # resolved by the bounded producer immediately before encoding.
                entries.append(dict(channel=channel, url=video_url, id=video_id))
            except Exception as exc:
                failures.append((str(entry.get('id') or url), f'{type(exc).__name__}: {exc}'))
    return entries, failures


def stream_folders(args, catalog):
    """Match provider channel identities to selected local folder names."""
    from src.archive.profiles import folder_aliases
    root = Path(args.output_root).resolve()
    aliases = {folder.name: folder_aliases(folder) for folder in sorted(root.iterdir())
               if folder.is_dir() and not folder.is_symlink() and not folder.name.startswith('.')
               and not folder.name.endswith('.uf')} if root.is_dir() else {}
    selected = getattr(args, '_selected_stream_folders', None)
    mapping = {}
    for channel in dict.fromkeys(item['channel'] for item in catalog):
        matches = [name for name, values in aliases.items() if channel in values
                   and (selected is None or name in selected)]
        if len(matches) > 1:
            raise ValueError(f'Ambiguous channel {channel}: {", ".join(matches)}; select a folder ID with --channels')
        mapping[channel] = matches[0] if matches else channel
    for item in catalog:
        item['folder'] = mapping[item['channel']]


def stream_selector_sources(args):
    from src.archive.profiles import channel_url, folder_aliases, resolve_selectors
    selectors = getattr(args, '_stream_selectors', None)
    if not selectors:
        return
    root = Path(args.output_root).resolve()
    pairs = resolve_selectors(root, selectors, args, allow_new=True, return_pairs=True)
    args._selected_stream_folders = list(dict.fromkeys(name for _, name in pairs))
    urls = []
    for selector, name in pairs:
        if selector.startswith(('http://', 'https://')):
            url = channel_url(selector)
            urls.append(url + ('/videos' if 'youtube.com/' in url else '/videos?filter=archives&sort=time'))
            continue
        aliases = folder_aliases(root / name) if (root / name).is_dir() else set()
        youtube_id = next((value for value in sorted(aliases | {name})
                           if re.fullmatch(r'UC[A-Za-z0-9_-]{22}', value)), None)
        if youtube_id:
            urls.append('https://www.youtube.com/channel/' + youtube_id + '/videos')
            continue
        provider_urls = sorted(value for value in aliases if value.startswith('https://twitch.tv/'))
        if provider_urls:
            url = provider_urls[0]
        else:
            if name.isdigit():
                raise ValueError(f'No Twitch URL saved for channel ID {name}; pass its channel URL')
            url = 'https://www.twitch.tv/' + safe_component(name)
        urls.append(url + '/videos?filter=archives&sort=time')
    args.source_urls = list(dict.fromkeys(urls))


def catalog_jobs(args, binary, counts, cancel_event):
    for item in args._stream_catalog:
        if cancel_event is not None and cancel_event.is_set():
            return
        video_id, video_url, channel = item['id'], item['url'], item['channel']
        folder = item['folder']
        profile = args._stream_profiles[folder]
        options = profile['options']
        try:
            event('stream_resolving', video_id=video_id)
            info = metadata(video_url, binary, args.cookies, cancel_event=cancel_event)
            if info.get('is_live') or info.get('live_status') == 'is_upcoming':
                counts['skipped'] = counts.get('skipped', 0) + 1
                event('stream_skipped_live', job_key=video_url, source=video_url, video_id=video_id)
                continue
            actual = info.get('channel_id') or info.get('uploader_id')
            if actual != channel:
                raise ValueError(f'Provider channel changed during discovery: {channel} -> {actual}')
            video, audio = select_formats(info, args.source_max_height)
            original = dict(width=int(video['width']), height=int(video['height']),
                            fps=float(video.get('fps') or info.get('fps') or 30),
                            duration=float(info.get('duration') or 0), audio=audio is not None)
            remote = dict(url=video_url, media=original, binary=binary, cookies=args.cookies,
                          video_format=video['format_id'], audio_format=audio['format_id'] if audio else None,
                          public_metadata=info, thumbnail=info.get('thumbnail'),
                          thumbnail_headers=info.get('http_headers'), write_thumbnail=args.write_thumbnail,
                          thumbnail_codec=options.thumbnail_codec, thumbnail_qp=options.thumbnail_qp)
            legacy = Path(args.output_root).resolve() / folder / (video_id + '.uf')
            final = profile['archive_urls'].get(video_url)
            if final is None:
                final = legacy.with_name(f'{video_id}_{upload_timestamp(info)}.uf.json')
            job = dict(source=video_id, relative=folder + '/' + video_id, final=str(final), remote=remote,
                       resume_check=getattr(args, 'resume_check', 'fast'),
                       hwaccel=args.ff_hwaccel, layout='flat', filename_style='rt', legacy=str(legacy),
                       lease_key=legacy.name, lease_seconds=args.shared_lease_seconds,
                       input_threads=options.input_threads, prefetch_frames=options.prefetch_frames,
                       _config=profile['config'], _paths=profile['paths'])
            event('stream_selected', video_id=video_id, channel_id=channel,
                  video_format=remote['video_format'], audio_format=remote['audio_format'], archive=str(final))
            yield job
        except Exception as exc:
            counts['failed'] += 1
            event('encode_failed', job_key=video_url, source=video_url, error=f'{type(exc).__name__}: {exc}')


def run_stream(args):
    from src.archive.profiles import Profiles
    from src.archive.coordinator import archived_job, check_completed
    from src.archive.stream_pipeline import execute_stream_jobs
    binary = shutil.which(args.ytdlp_bin)
    if not binary:
        raise ValueError('yt-dlp not found; pass --yt-dlp /path/to/yt-dlp')
    args._cleanup_sources = set()
    stream_selector_sources(args)
    catalog, discovery_failures = stream_catalog(args, binary)
    stream_folders(args, catalog)
    profiles = Profiles(args)
    args._stream_profiles = {}
    existing = []
    # Every discovered channel is preflighted before any video worker starts.
    selected_channels = list(dict.fromkeys([item['folder'] for item in catalog]
                                          + getattr(args, '_selected_stream_folders', [])))
    for channel in selected_channels:
        profile = profiles.load(Path(args.output_root).resolve() / channel)
        if profile['config']['max_frames'] is not None:
            raise ValueError(f'{channel}: streaming requires full videos; omit --max_frames')
        urls = {}
        for path, data in profile['archives']:
            url = data['source'].get('url')
            if url:
                if url in urls:
                    raise ValueError(f'Multiple archives for {url} in {profile["root"]}')
                urls[url] = path if path.name.endswith('.uf.json') else path.parent
            existing.append((profile, archived_job(path, data, profile['options'])))
        profile['archive_urls'] = urls
        args._stream_profiles[channel] = profile
    # Lifecycle and explicit cleanup stay scoped to the selected channels.
    args.channel_ids = list(args._stream_profiles)
    args._channel_profiles = {str(profile['root']): profile['paths'] for profile in args._stream_profiles.values()}
    archived_urls = {url for profile in args._stream_profiles.values() for url in profile['archive_urls']}
    pending = [item for item in catalog if item['url'] not in archived_urls]
    if args.max_videos:
        pending = pending[:args.max_videos]
    args._stream_catalog = pending
    from copy import copy
    from src.archive.thumbnails import discover as discover_images, run_pass
    thumbnail_passes = []
    for channel, profile in args._stream_profiles.items():
        options = copy(profile['options'])
        options.base_root, options.output_root = args.output_root, args.output_root
        options.channel_ids, options.input_file = [channel], None
        options._cleanup_sources = args._cleanup_sources
        options._thumbnail_jobs = discover_images(options) if options.thumbnail_codec == 'dcvc-intra' else []
        thumbnail_passes.append((options, profile))
    image_count = sum(len(options._thumbnail_jobs) for options, _ in thumbnail_passes)
    event('run_discovered', total=len(existing) + len(pending) + len(discovery_failures) + image_count)
    counts = {'encoded':0,'resumed':0,'busy':0,'failed':0}
    failed = set()
    if existing:
        event('resume_check_started', total=len(existing),
              resume_check=getattr(args, 'resume_check', 'fast'),
              message=f'Checking {len(existing):,} existing video archives before streaming remaining videos')
        for profile in args._stream_profiles.values():
            jobs = [job for owner, job in existing if owner is profile]
            for job, status in zip(jobs, check_completed(jobs, profile['config'], args.procs)):
                counts[status] += 1
                if status == 'failed':
                    failed.add(str(Path(job['final']).parent))
        event('resume_check_completed', total=len(existing), resumed=counts['resumed'], failed=counts['failed'])
    for key, error in discovery_failures:
        counts['failed'] += 1
        event('encode_failed', job_key='discovery:' + key, source=key, error=error)
    outcomes,worker_failures = execute_stream_jobs(args,None,None,binary,counts)
    failed.update(worker_failures)
    for result in outcomes: counts[result if result in counts else 'failed'] += 1
    for options, profile in thumbnail_passes:
        failed.update(run_pass(options, profile['config'], profile['paths']))
    event('stream_finished',**counts)
    from src.archive.lifecycle import finish
    from src.archive.dashboard import command_queue
    finish(args,command_queue(),failed)
    return int(counts['failed'] > 0 or bool(failed))


def store_thumbnail(remote, output_base):
    """Keep RT's thumbnail name and never replace a peer's completed sidecar."""
    import urllib.parse
    import urllib.request
    url = remote.get('thumbnail')
    if not isinstance(url,str) or not url.startswith(('https://','http://')): return None
    extension = Path(urllib.parse.urlparse(url).path).suffix.lower()
    if extension not in ('.jpg','.jpeg','.png','.webp'): extension = '.jpg'
    target = output_base.with_name(output_base.name+extension)
    if target.exists():
        if target.is_symlink(): raise ValueError('Unsafe thumbnail path')
        return target
    with tempfile.TemporaryDirectory(prefix='.uf-thumbnail-',dir=target.parent) as temporary:
        stage = Path(temporary)/target.name
        request = urllib.request.Request(url,headers=remote.get('thumbnail_headers') or {})
        with urllib.request.urlopen(request,timeout=30) as response, stage.open('xb') as output:
            shutil.copyfileobj(response,output);output.flush();os.fsync(output.fileno())
        from PIL import Image
        with Image.open(stage) as image: image.verify()
        try: os.link(stage,target)
        except FileExistsError: pass
    return target


def finish_thumbnail(remote,final,config,paths,device,instance,codec=None):
    if not remote or not remote.get('write_thumbnail'): return
    try:
        image = store_thumbnail(remote,Path(final).with_name(Path(final).name.removesuffix('.uf.json')))
        if image and remote.get('thumbnail_codec') == 'dcvc-intra':
            from copy import copy
            from src.archive.thumbnails import encode_images
            qp = remote['thumbnail_qp']
            image_codec = None
            if codec is not None:
                image_codec = copy(codec);image_codec.video = None
                if config.get('runtime')=='int16':
                    image_codec.preamble = image_codec.preamble[:40]+bytes(32)
            encode_images([(image,image.with_name(f'{image.name}_qI{qp}.dcvci'))],
                          {**config,'qp_i':qp,'qp_p':qp},paths,device,instance,codec=image_codec,track_progress=False)
    except Exception as exc:
        event('thumbnail_failed',source=str(final),error=str(exc))


def safe_component(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}',value):
        raise ValueError(f'Invalid channel/video identifier: {value!r}')
    return value


def channel_urls(youtube,twitch):
    urls = []
    for channel in youtube:
        if channel.lower().endswith('.txt'):
            for line in Path(channel).read_text().splitlines():
                value = line.strip()
                if not value or value.startswith('#'): continue
                urls.append('https://www.youtube.com/watch?v='+safe_id(value))
        else:
            urls.append('https://www.youtube.com/channel/'+safe_id(channel,channel=True)+'/videos')
    urls.extend('https://www.twitch.tv/'+safe_component(channel)+'/videos?filter=archives&sort=time' for channel in twitch)
    return urls
