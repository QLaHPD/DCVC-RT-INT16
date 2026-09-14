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


def select_formats(info, max_height=0):
    formats = [f for f in info.get('formats', []) if f.get('url') and f.get('vcodec') not in (None, 'none')
               and f.get('height') and not f.get('has_drm')]
    if not formats:
        raise ValueError('No downloadable video format')
    if max_height:
        bounded = [f for f in formats if f['height'] <= max_height]
        formats = bounded or [f for f in formats if f['height'] == min(v['height'] for v in formats)]
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


def existing_stream_archives(channel_out):
    """One metadata snapshot per channel, preserving prior naming on resume."""
    archives = {}
    for path in Path(channel_out).glob('*.uf.json'):
        data = json.loads(path.read_text())
        url = data.get('source',{}).get('url')
        if data.get('format') != 'dcvc-uf-archive' or not url: continue
        if url in archives:
            raise ValueError(f'Multiple archives for {url}: {archives[url]} and {path}')
        archives[url] = path
    return archives

def stream_jobs(args, binary, counts):
    from src.archive.media import decoder_thread_budget
    seen = set()
    archives = {}
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
                youtube = 'twitch.tv' not in url.lower()
                video_id = safe_id(entry.get('id')) if youtube else safe_component(entry.get('id'))
                if video_id in seen:
                    continue
                if getattr(args,'max_videos',None) and len(seen) >= args.max_videos:
                    return
                seen.add(video_id)
                event('run_discovered', total=len(seen)+len(pending))
                video_url = 'https://www.youtube.com/watch?v=' + video_id if youtube else (entry.get('webpage_url') or entry.get('url') or 'https://www.twitch.tv/videos/'+video_id.lstrip('v'))
                event('stream_resolving', video_id=video_id)
                info = metadata(video_url, binary, args.cookies)
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
                if channel not in archives:
                    archives[channel] = existing_stream_archives(legacy.parent)
                final = archives[channel].get(video_url)
                if final is None:
                    final = legacy.with_name(f'{video_id}_{upload_timestamp(info)}.uf.json')
                job = dict(source=video_id, relative=channel+'/'+video_id, final=str(final), remote=remote,
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


def run_stream(args):
    from src.archive.workflow import pipeline, execute_jobs
    if args.max_frames:
        raise ValueError('Streaming requires full videos; omit --max_frames')
    binary = shutil.which(args.ytdlp_bin)
    if not binary:
        raise ValueError('yt-dlp not found; pass --yt-dlp /path/to/yt-dlp')
    args._cleanup_sources = set()
    config,paths = pipeline(args)
    counts = {'encoded':0,'resumed':0,'busy':0,'failed':0}
    outcomes,failed = execute_jobs(args,stream_jobs(args,binary,counts),config,paths)
    for result in outcomes: counts[result if result in counts else 'failed'] += 1
    event('stream_finished',**counts)
    from src.archive.lifecycle import finish
    from src.archive.dashboard import command_queue
    finish(args,command_queue(),failed)
    return int(counts['failed'] > 0)


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
                          {**config,'qp_i':qp,'qp_p':qp},paths,device,instance,codec=image_codec)
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
