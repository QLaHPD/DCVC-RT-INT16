"""Recover channel profiles without inheriting another machine's paths."""

from copy import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
from urllib.parse import urlsplit

from src.archive.storage import event, sha256


FIELDS = {
    'runtime': 'runtime', 'model_structure': 'variant', 'qp_i': 'qp_i', 'qp_p': 'qp_p',
    'fps': 'fps', 'resolution': 'resolution', 'reset_interval': 'reset_interval',
    'intra_period': 'intra_period', 'skip_thres': 'skip_threshold', 'max_frames': 'max_frames',
    'audio': 'audio', 'opus_channels': 'opus_channels', 'opus_bitrate': 'opus_bitrate',
    'opus_frame_ms': 'opus_frame_ms', 'opus_complexity': 'opus_complexity', 'opus_vbr': 'opus_vbr',
}
DEFAULTS = dict(runtime='fp16', skip_threshold=0, opus_frame_ms=20,
                opus_complexity=10, opus_vbr='on')


def canonical(config):
    return json.dumps(config, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def read_json(path, retry=False):
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f'Saved metadata must not be a symlink: {path}')
    for attempt in range(50 if retry else 1):
        try:
            with path.open() as source:
                value = json.load(source)
            if not isinstance(value, dict):
                raise ValueError('expected a JSON object')
            return value
        except json.JSONDecodeError as exc:
            if not retry or attempt == 49:
                raise ValueError(f'Malformed saved metadata {path}: {exc}') from exc
            time.sleep(.02)
    raise ValueError(f'Cannot read saved metadata: {path}')


def validate_config(config, where, image=False):
    """Reject incomplete profiles rather than silently filling required fields."""
    if not isinstance(config, dict):
        raise ValueError(f'Invalid UF profile in {where}')
    required = {'codec', 'variant', 'image_sha256', 'video_sha256', 'qp_i', 'qp_p',
                'max_frames', 'intra_period', 'audio'}
    if not image:
        required |= {'archive_version', 'fps', 'resolution', 'reset_interval',
                     'opus_channels', 'opus_bitrate'}
    missing = required - config.keys()
    if missing:
        raise ValueError(f'Incomplete UF profile in {where}: missing {", ".join(sorted(missing))}')
    if (config['codec'] != 'dcvc-uf' or config.get('runtime', 'fp16') not in ('int16', 'fp16')
            or config.get('archive_version', 1) != 1
            or config['variant'] not in (('image',) if image else ('hts', 'htl', 'ld'))
            or bool(config.get('image_only')) != image):
        raise ValueError(f'Unsupported UF profile in {where}')
    allowed = set(FIELDS.values()) | {'codec', 'archive_version', 'image_sha256',
                                     'video_sha256', 'integer', 'image_only'}
    if config.keys() - allowed:
        raise ValueError(f'Unknown UF profile fields in {where}: {sorted(config.keys() - allowed)}')
    def integer(key, low, nullable=False):
        value = config.get(key)
        if nullable and value is None:
            return
        if type(value) is not int or value < low:
            raise ValueError(f'Invalid {key} in {where}: {value!r}')
    for key in ('qp_i', 'qp_p'):
        integer(key, 0)
        if config[key] > 63:
            raise ValueError(f'Invalid {key} in {where}')
    integer('max_frames', 1, nullable=True)
    integer('intra_period', -1)
    if config['intra_period'] == 0 or (config['variant'] in ('hts', 'htl')
            and config['intra_period'] > 1 and config['intra_period'] % 8):
        raise ValueError(f'Invalid intra_period in {where}')
    if not image:
        integer('resolution', 1, nullable=True)
        integer('reset_interval', -1)
        fps = config['fps']
        if fps is not None and (type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0):
            raise ValueError(f'Invalid fps in {where}')
    if config['audio'] not in ('opus', 'none'):
        raise ValueError(f'Invalid audio in {where}')
    if not image and (config['opus_channels'] not in ('mono', 'stereo')
            or not isinstance(config['opus_bitrate'], str)
            or not config['opus_bitrate'].strip()):
        raise ValueError(f'Invalid Opus settings in {where}')
    if (config.get('opus_frame_ms', 20) not in (2.5, 5, 10, 20, 40, 60)
            or type(config.get('opus_complexity', 10)) is not int
            or not 0 <= config.get('opus_complexity', 10) <= 10
            or config.get('opus_vbr', 'on') not in ('on', 'off', 'constrained')):
        raise ValueError(f'Invalid Opus settings in {where}')
    skip = config.get('skip_threshold', 0)
    if type(skip) not in (int, float) or not math.isfinite(skip) or not 0 <= skip <= 16:
        raise ValueError(f'Invalid skip_threshold in {where}')
    hashes = [config['image_sha256']]
    if not image:
        hashes.append(config['video_sha256'])
    if config.get('runtime', 'fp16') == 'int16':
        identity = config.get('integer', {})
        if not isinstance(identity, dict) or set(identity) != {'arithmetic', 'image', 'video'}:
            raise ValueError(f'Missing integer identities in {where}')
        hashes.extend((identity['image'], identity['video']))
        if not isinstance(identity['arithmetic'], str) or not identity['arithmetic']:
            raise ValueError(f'Missing integer arithmetic in {where}')
    if any(not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value) for value in hashes):
        raise ValueError(f'Invalid model identities in {where}')
    return config


def channel_metadata(root, recursive=True):
    """Walk visible channel directories; coordination and payload trees are pruned."""
    root = Path(root)
    if not root.exists():
        return
    for directory, dirs, files in os.walk(root, followlinks=False):
        directory = Path(directory)
        profile = directory / '.dcvc-uf-shared-work/pipeline.json'
        if profile.exists() or profile.is_symlink():
            yield 'profile', profile
        image = directory / '.dcvc-uf-image-work/pipeline.json'
        if image.exists() or image.is_symlink():
            yield 'image', image
        for name in sorted(files):
            if not name.startswith('.') and name.endswith('.uf.json'):
                yield 'archive', directory / name
        for name in sorted(dirs):
            if name.endswith('.uf') and not name.startswith('.'):
                manifest = directory / name / 'manifest.json'
                if manifest.is_file():
                    yield 'archive', manifest
        dirs[:] = sorted(name for name in dirs if recursive and not name.startswith('.')
                         and not name.endswith('.uf') and not (directory / name).is_symlink())


def saved_profile(root, required=False, recursive=True):
    profiles, archives, images = [], [], []
    for kind, path in channel_metadata(root, recursive):
        value = read_json(path, retry=kind in ('profile', 'image'))
        if kind == 'archive':
            if value.get('format') != 'dcvc-uf-archive':
                continue
            if value.get('version') != 1:
                raise ValueError(f'Unsupported UF archive: {path}')
            config = validate_config(value.get('pipeline'), path)
            archives.append((path, value))
        else:
            config = validate_config(value.get('config'), path, image=kind == 'image')
            digest = hashlib.sha256(canonical(config).encode()).hexdigest()
            if value.get('schema_version') != 1 or value.get('pipeline_id') != digest:
                raise ValueError(f'Invalid saved pipeline identity: {path}')
        (images if kind == 'image' else profiles).append((config, path))
    def consistent(values, label):
        if not values:
            return None
        first, origin = values[0]
        for config, path in values[1:]:
            if config != first:
                raise ValueError(f'Conflicting {label} settings for {root}: {origin} and {path}; '
                                 + difference(first, config))
        return first
    config = consistent(profiles, 'UF video')
    image = consistent(images, 'UF thumbnail')
    if required and config is None:
        raise ValueError(f'No saved UF video configuration in {root}; use encode with explicit settings first')
    return config, unique_archives(archives), image


def unique_archives(archives):
    """Prefer a migrated flat commit over its identical legacy bundle."""
    selected = {}
    for path, data in archives:
        source = data.get('source')
        if not isinstance(source, dict) or not isinstance(source.get('path'), str):
            raise ValueError(f'Invalid UF source metadata: {path}')
        flat = path.name.endswith('.uf.json')
        parent = path.parent if flat else path.parent.parent
        key = (parent, source.get('url') or source.get('relative_path') or Path(source['path']).name)
        previous = selected.get(key)
        if previous:
            old_path, old = previous
            payloads = lambda value: {name: record for name, record in value.get('artifacts', {}).items()
                                      if name in ('video.bin', 'audio.opus')}
            if (old_path.name.endswith('.uf.json') == flat or payloads(old) != payloads(data)
                    or old['source'].get('sha256') != source.get('sha256')):
                raise ValueError(f'Ambiguous archives for one video: {old_path} and {path}')
            if not flat:
                continue
        selected[key] = (path, data)
    return list(selected.values())


def difference(saved, requested):
    return '; '.join(f'{key}: saved={saved.get(key, DEFAULTS.get(key))!r}, '
                     f'requested={requested.get(key, DEFAULTS.get(key))!r}'
                     for key in sorted(saved.keys() | requested.keys())
                     if saved.get(key, DEFAULTS.get(key)) != requested.get(key, DEFAULTS.get(key)))


def channel_url(value):
    parsed = urlsplit(value)
    host = (parsed.hostname or '').lower().removeprefix('www.').removeprefix('m.')
    if parsed.scheme not in ('https', 'http') or host not in ('youtube.com', 'twitch.tv'):
        raise ValueError(f'Expected a YouTube or Twitch channel URL: {value}')
    parts = parsed.path.strip('/').split('/')
    if host == 'youtube.com':
        if parts[0].startswith('@'):
            path = parts[0].lower()
        elif len(parts) >= 2 and parts[0] in ('channel', 'c', 'user'):
            path = '/'.join(parts[:2])
        else:
            raise ValueError(f'Expected a channel URL, not a video/playlist URL: {value}')
    else:
        if not parts[0] or parts[0] in ('videos', 'directory', 'downloads'):
            raise ValueError(f'Expected a Twitch channel URL: {value}')
        path = parts[0].lower()
    return f'https://{host}/{path}'


def folder_aliases(folder):
    aliases = {folder.name}
    candidates = sorted(folder.glob('*.info.json'))
    state = folder / '.channel-state.json'
    if state.is_file():
        candidates.insert(0, state)
    # Read one document at a time; avoid retaining full provider format lists.
    for path in candidates[:32]:
        if path.is_symlink():
            continue
        try:
            data = read_json(path)
            for key in ('channel_id', 'uploader_id', 'channel_url', 'uploader_url'):
                value = data.get(key)
                if isinstance(value, str):
                    aliases.add(channel_url(value) if value.startswith(('http://', 'https://')) else value)
        except (OSError, ValueError):
            continue
    return aliases


def resolve_selectors(root, selectors, args, allow_new=False, return_pairs=False):
    from src.archive.streaming import safe_component, metadata
    root = Path(root).resolve()
    folders = [path for path in sorted(root.iterdir()) if path.is_dir()
               and not path.is_symlink() and not path.name.startswith('.') and not path.name.endswith('.uf')] if root.is_dir() else []
    aliases = None
    names = []
    pairs = []
    for selector in selectors:
        if not selector.startswith(('http://', 'https://')):
            name = safe_component(selector)
        else:
            normalized = channel_url(selector)
            if aliases is None:
                aliases = {path.name: folder_aliases(path) for path in folders}
            matches = [name for name, values in aliases.items() if normalized in values]
            if len(matches) > 1:
                raise ValueError(f'Ambiguous channel URL {selector}: {", ".join(matches)}; pass the folder ID')
            if matches:
                name = matches[0]
            else:
                # /channel/UC... needs no provider request.
                parts = urlsplit(normalized).path.strip('/').split('/')
                name = parts[1] if len(parts) == 2 and parts[0] == 'channel' else None
                if name is None or not (root / name).is_dir():
                    binary = shutil.which(args.ytdlp_bin)
                    if not binary:
                        raise ValueError(f'Cannot resolve {selector} locally; install yt-dlp or pass the channel folder ID')
                    info = metadata(selector, binary, args.cookies, flat=True)
                    name = safe_component(info.get('channel_id') or info.get('uploader_id'))
                    matches = [folder for folder, values in aliases.items() if name in values]
                    if len(matches) > 1:
                        raise ValueError(f'Ambiguous channel {selector}: {", ".join(matches)}; pass the folder ID')
                    if matches:
                        name = matches[0]
        safe_component(name)
        if not allow_new and not (root / name).is_dir():
            raise ValueError(f'Channel directory does not exist: {root / name}')
        if (root / name).is_symlink():
            raise ValueError(f'Channel directory must not be a symlink: {root / name}')
        if name not in names:
            names.append(name)
        pairs.append((selector, name))
    return pairs if return_pairs else names


def normalize_inputs(args):
    """Normalize root aliases before starting the UI; resolve selectors in the backend."""
    root = getattr(args, 'root', None)
    if getattr(args, 'channels', None) and (args.youtube_channels or args.twitch_channels):
        raise ValueError('select either --channels or --youtube_channels/--twitch_channels')
    if root:
        if args.input_file:
            raise ValueError('--root cannot be combined with --input_file')
        if args.base_root and Path(args.base_root).resolve() != Path(root).resolve():
            raise ValueError('--root and --base_root refer to different folders')
        if not args.source_urls and not args.youtube_channels and not args.twitch_channels and args.command != 'stream':
            args.base_root = root
        if args.output_root and Path(args.output_root).resolve() != Path(root).resolve():
            raise ValueError('--root and --output_root refer to different folders; use --base_root and --output_root for separate paths')
        args.output_root = root
    if not args.output_root:
        raise ValueError('provide --root or --output_root')
    if args.command == 'join':
        if args.source_urls or args.youtube_channels or args.twitch_channels or args.input_file:
            raise ValueError('join processes local originals; select channels with --channels or --channel_ids')
        if not args.base_root:
            raise ValueError('join requires --root or --base_root')
        if args.auto_delete or args.cleanup_dry_run:
            raise ValueError('join retains originals; cleanup is a separate command')
        args.keep_originals = True
    selectors = list(args.channel_ids or []) + list(getattr(args, 'channels', None) or [])
    if selectors and args.input_file:
        raise ValueError('--channels/--channel_ids cannot be combined with --input_file')
    if selectors and (args.command == 'stream' or args.source_urls):
        if args.source_urls:
            raise ValueError('select either --channels/--channel_ids or --source_urls for streaming')
        args._stream_selectors = selectors
        args.source_urls = [value if value.startswith(('http://', 'https://')) else
                            ('https://www.youtube.com/channel/' + value + '/videos'
                             if value.startswith('UC') else 'https://www.twitch.tv/' + value + '/videos')
                            for value in selectors]
        args.channel_ids = None
        args.channels = None
    elif selectors:
        args.channel_ids = selectors
    if args.command == 'stream' and args.base_root:
        raise ValueError('stream uses remote inputs; use encode or join for --base_root')


def validate_options(args):
    from src.archive.media import decoder_thread_budget
    if args.input_threads is None:
        args.input_threads = decoder_thread_budget(args.procs) if args.runtime == 'int16' else 1
    if args.prefetch_frames is None:
        args.prefetch_frames = 8 if args.runtime == 'int16' else 0
    if args.device == 'cpu' and args.runtime != 'int16':
        raise ValueError('--device cpu requires an INT16 channel')
    for name in ('resolution', 'max_frames', 'procs', 'input_threads'):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            raise ValueError(f'--{name} must be positive')
    if args.prefetch_frames < 0:
        raise ValueError('--prefetch_frames must be nonnegative')
    if args.intra_period == 0 or args.intra_period < -1:
        raise ValueError('--intra_period must be -1 or positive')
    if args.model_structure != 'ld' and args.intra_period > 1 and args.intra_period % 8:
        raise ValueError('--intra_period for HT-S/HT-L must be 1 or a multiple of 8')
    if args.reset_interval < -1:
        raise ValueError('--reset_interval must be -1, 0 or positive')


class Profiles:
    """Checked per-channel options and local paths, with bounded model-header reuse."""
    def __init__(self, args):
        self.args = args
        self.models = {}
        self.channels = {}

    def model_paths(self, config, options):
        from src.archive.workflow import model_paths, ROOT
        paths = model_paths(config['variant'], options.model_path_i, options.model_path_p)
        image_only = config.get('image_only', False)
        if config.get('runtime', 'fp16') == 'fp16':
            for path, key in zip(paths[:1 if image_only else 2], ('image_sha256', 'video_sha256')):
                signature = (path, Path(path).stat())
                if self.models.get(signature) != config[key]:
                    if sha256(path) != config[key]:
                        raise ValueError(f'Checkpoint {path} does not match saved {key}={config[key]}')
                    self.models[signature] = config[key]
            return paths
        from src.int16.prepared import load_prepared
        from src.int16.ops import ARITHMETIC_ID
        if config['integer']['arithmetic'] != ARITHMETIC_ID:
            raise ValueError(f'Unsupported saved integer arithmetic: {config["integer"]["arithmetic"]}')
        prepared = []
        for path, override, key in zip(paths, (options.model_path_i, options.model_path_p),
                                       ('image_sha256', 'video_sha256')):
            if override and Path(path).is_file() and (not image_only or key == 'image_sha256'):
                if sha256(path) != config[key]:
                    raise ValueError(f'Explicit checkpoint {path} does not match saved {key}={config[key]}')
        for source, variant, override, kind in zip(paths, ('image', config['variant']),
                (options.prepared_i, options.prepared_p), ('image', 'video')):
            if image_only and kind == 'video':
                prepared.append(None)
                break
            expected = config['integer'][kind]
            source_hash = config[kind + '_sha256']
            if override:
                candidates = [Path(override)]
            else:
                canonical_path = Path(source).with_name(f'{Path(source).name}.{ARITHMETIC_ID}.{source_hash[:16]}.int16.pt')
                candidates = [canonical_path]
                for directory in dict.fromkeys((Path(source).parent, ROOT / 'checkpoints')):
                    candidates.extend(sorted(directory.glob('*.int16.pt')))
            selected, errors = None, []
            for candidate in dict.fromkeys(candidates):
                if not candidate.is_file():
                    continue
                stat = candidate.stat()
                key = (str(candidate.resolve()), stat.st_dev, stat.st_ino, stat.st_size,
                       stat.st_mtime_ns, stat.st_ctime_ns)
                known = self.models.get(key)
                try:
                    if known is None:
                        data = load_prepared(candidate, memory_map=True)
                        known = (data['variant'], data['source_sha256'], data['identity'])
                        del data
                        self.models[key] = known
                    if known != (variant, source_hash, expected):
                        raise ValueError('model identity/variant differs from saved configuration')
                    selected = str(candidate.resolve())
                    break
                except (ValueError, OSError, RuntimeError) as exc:
                    errors.append(f'{candidate}: {exc}')
                    if override:
                        raise ValueError(f'--prepared_{"i" if kind == "image" else "p"}: {errors[-1]}') from exc
            if selected is None:
                raise ValueError(f'Missing matching local prepared {kind} model {expected}; copy the matching '
                                 f'*.int16.pt into checkpoints/ or pass --prepared_{"i" if kind == "image" else "p"}'
                                 + ('; ' + errors[-1] if errors else ''))
            prepared.append(selected)
        return paths + tuple(prepared)

    def load(self, root, required=False, image_only=False):
        root = Path(root).resolve()
        if root in self.channels:
            return self.channels[root]
        saved, archives, image = saved_profile(root, required=required, recursive=self.args.recursive)
        options = copy(self.args)
        explicit = getattr(options, '_explicit_codec_options', set())
        if saved:
            for field, key in FIELDS.items():
                if field not in explicit:
                    setattr(options, field, saved.get(key, DEFAULTS.get(key)))
        elif image and image_only:
            for field, key in (('runtime', 'runtime'), ('skip_thres', 'skip_threshold')):
                if field not in explicit:
                    setattr(options, field, image.get(key, DEFAULTS.get(key)))
        from src.archive.presets import apply_target_psnr
        apply_target_psnr(options)
        if saved:
            requested = dict(saved)
            for field, key in FIELDS.items():
                requested[key] = getattr(options, field)
            mismatches = difference(saved, requested)
            if mismatches:
                raise ValueError(f'Encoding settings differ for channel {root.name}: {mismatches}')
        validate_options(options)
        if saved:
            config = saved
            try:
                paths = self.model_paths(config, options)
            except (ValueError, OSError) as exc:
                raise ValueError(f'Channel {root.name}: {exc}') from exc
        elif image and image_only:
            requested = {**image, 'runtime': options.runtime}
            if options.skip_thres:
                requested['skip_threshold'] = options.skip_thres
            else:
                requested.pop('skip_threshold', None)
            if difference(image, requested):
                raise ValueError(f'Thumbnail settings differ for channel {root.name}: {difference(image, requested)}')
            config = image
            paths = self.model_paths(config, options)
        else:
            from src.archive.workflow import pipeline
            config, paths = pipeline(options, image_only=image_only)
        if image:
            if 'thumbnail_codec' in explicit and options.thumbnail_codec != 'dcvc-intra':
                raise ValueError(f'Thumbnail settings differ for channel {root.name}: saved dcvc-intra')
            if 'thumbnail_qp' in explicit and options.thumbnail_qp != image['qp_i']:
                raise ValueError(f'Thumbnail QP differs for channel {root.name}: saved={image["qp_i"]}, requested={options.thumbnail_qp}')
            options.thumbnail_codec = 'dcvc-intra'
            options.thumbnail_qp = image['qp_i']
            from src.archive.thumbnails import image_pipeline
            expected = image_pipeline({**config, 'qp_i': options.thumbnail_qp})
            if expected != image:
                raise ValueError(f'Thumbnail models/settings differ for channel {root.name}: {difference(image, expected)}')
        result = dict(root=root, options=options, config=config, paths=paths, archives=archives)
        self.channels[root] = result
        if saved:
            event('channel_profile_loaded', channel=str(root), runtime=config.get('runtime', 'fp16'),
                  message=f'{root.name}: inherited saved UF {config["variant"].upper()} configuration')
        if getattr(options, 'target_psnr_selection', None):
            event('target_psnr_selected', channel=str(root), **options.target_psnr_selection)
        return result
