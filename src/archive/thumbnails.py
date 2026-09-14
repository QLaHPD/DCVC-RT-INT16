"""UF intra thumbnails with RT names, self-describing files and exact-path cleanup."""
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile

import numpy as np
from PIL import Image, ImageOps

from src.archive.storage import SharedWorkPool, Heartbeat, identity, sha256, sync_directory, event, has_completion_record, memory_stage_root, completed_disk_stage

MAGIC = b'UFIMAGE\x01'
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff', '.avif'}


class ImagePool(SharedWorkPool):
    @staticmethod
    def root(channel_out):
        return Path(channel_out) / '.dcvc-uf-image-work'


def read_image(path, verify_payload=True):
    with open(path, 'rb') as stream:
        if stream.read(8) != MAGIC:
            raise ValueError('Not a UF intra image (RT and UF bitstreams use different networks)')
        length_bytes = stream.read(4)
        if len(length_bytes) != 4:
            raise ValueError('Truncated UF image header')
        length = struct.unpack('>I', length_bytes)[0]
        if not 0 < length <= 1024*1024:
            raise ValueError('Invalid UF image header length')
        metadata = json.loads(stream.read(length))
        if metadata.get('format') != 'dcvc-uf-image' or not metadata['pipeline'].get('image_only'):
            raise ValueError('Invalid UF image metadata')
        if not has_completion_record(metadata):
            raise ValueError('Intra image has no recognized completion record')
        if metadata['video']['frames'] != 1 or metadata['video']['packets'] != 1:
            raise ValueError('Intra image must contain exactly one frame')
        offset = stream.tell()
        if not verify_payload:
            return metadata, offset
        digest = hashlib.sha256()
        size = 0
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk); size += len(chunk)
        if digest.hexdigest() != metadata['payload_sha256'] or size != metadata['payload_size']:
            raise ValueError('UF image payload is incomplete or modified')
    return metadata, offset


class ImageReader:
    def __init__(self, path):
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert('RGB')
            rgb = np.asarray(image).astype(np.float32)
        self.original_height, self.original_width = rgb.shape[:2]
        r, g, b = rgb.transpose(2,0,1)
        y = .2126*r + .7152*g + .0722*b
        u = (b-y)/1.8556 + 128
        v = (r-y)/1.5748 + 128
        self.frame = np.stack((y,u,v)).clip(0,255).round().astype(np.uint8)
        self.frame = np.pad(self.frame, ((0,0),(0,self.original_height%2),(0,self.original_width%2)), mode='edge')
        _,self.height,self.width = self.frame.shape

    def read(self):
        frame,self.frame = self.frame,None
        return frame


def discover(args):
    base = Path(args.base_root).resolve() if args.base_root else Path(args.input_file).absolute().parent
    if args.input_file:
        sources = [Path(args.input_file).absolute()] if Path(args.input_file).suffix.lower() in EXTENSIONS else []
    else:
        sources = []
        for root in [base/n for n in args.channel_ids] if args.channel_ids else [base]:
            if not root.resolve().is_relative_to(base):
                raise ValueError('Channel escapes base_root')
            sources.extend(root.rglob('*') if args.recursive else root.iterdir())
    return [(p, Path(args.output_root).resolve()/p.relative_to(base).parent/f'{p.name}_qI{args.thumbnail_qp}.dcvci')
            for p in sorted(set(sources)) if p.is_file() and not p.is_symlink() and p.suffix.lower() in EXTENSIONS
            and not any(part.startswith('.') or part.endswith('.uf') for part in p.relative_to(base).parts)]



def image_pipeline(config):
    # Video model, FPS, audio and temporal QPs cannot alter an intra thumbnail.
    # This also permits resume after the channel's original videos were deleted.
    image = {key:config[key] for key in ('codec','runtime','image_sha256') if key in config}
    image.update(image_only=True,variant='image',video_sha256=None,qp_i=config['qp_i'],qp_p=config['qp_i'],
                 audio='none',max_frames=None,intra_period=1)
    if config.get('skip_threshold'): image['skip_threshold']=config['skip_threshold']
    if config.get('integer'):
        image['integer']={**config['integer'],'video':'0'*64}
    return image

def encode_images(jobs, config, paths, device, instance=None, codec=None, source_base=None):
    from src.archive.workflow import make_codec
    config = image_pipeline(config)
    failed = set()
    for source, target in jobs:
        source,target = Path(source),Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.environ['UF_PROGRESS_LOG'] = str(target.parent/'.thumbnail-progress.jsonl')
        os.environ['UF_JOB_KEY'] = str(target)
        pool = ImagePool(instance_name=instance)
        pool.ensure_pipeline(target.parent,config)
        lease,_ = pool.try_acquire(target.parent,target.name)
        if lease is None:
            failed.add(str(target.parent)); event('claimed_by_peer',source=str(source)); continue
        try:
            with Heartbeat(lease) as pulse:
                before, digest = identity(source),sha256(source)
                if target.exists():
                    old,_ = read_image(target)
                    if old['source']['sha256'] != digest or old['pipeline'] != config:
                        raise ValueError('Existing intra image has different source or configuration')
                    event('already_encoded',source=str(source),archive=str(target)); continue
                if codec is None:
                    codec = make_codec(config,paths,device)
                reader = ImageReader(source)
                event('encode_started',source=str(source),message=f'{reader.original_width}x{reader.original_height} -> {reader.original_width}x{reader.original_height} (intra image)')
                with tempfile.TemporaryDirectory(prefix='uf-image-',dir=memory_stage_root()) as temporary:
                    payload = Path(temporary)/'intra.bin'
                    result = codec.encode(reader,payload,config['qp_i'],config['qp_i'],intra_period=1,max_frames=1)
                    result.update(width=reader.width,height=reader.height,fps=1)
                    validation = {'mode': 'artifact-hashes', 'decode_performed': False}
                    metadata = dict(format='dcvc-uf-image',version=1,pipeline=config,video=result,validation=validation,
                                    image_width=reader.original_width,image_height=reader.original_height,
                                    source=dict(path=str(source),relative_path=str(source.relative_to(source_base)) if source_base else source.name,sha256=digest,**before),
                                    payload_sha256=sha256(payload),payload_size=payload.stat().st_size)
                    header = json.dumps(metadata,sort_keys=True).encode()
                    stage = Path(temporary)/target.name
                    with stage.open('xb') as output, payload.open('rb') as data:
                        output.write(MAGIC+struct.pack('>I',len(header))+header)
                        import shutil
                        shutil.copyfileobj(data,output); output.flush(); os.fsync(output.fileno())
                    if identity(source)!=before or sha256(source)!=digest:
                        raise ValueError('Thumbnail changed during encoding')
                    pulse.check()
                    payload.unlink()
                    disk_stage = completed_disk_stage(temporary, target, pulse)
                    try:
                        pulse.check()
                        os.link(disk_stage/target.name,target)  # Never replace a peer output.
                        sync_directory(target.parent)
                    finally:
                        shutil.rmtree(disk_stage)
                event('encode_completed',source=str(source),archive=str(target),**result)
        except Exception as exc:
            failed.add(str(target.parent)); event('encode_failed',source=str(source),error=str(exc))
        finally:
            lease.release()
    return failed


def decode_image(args, verify_only=False):
    from src.archive.frame_source import FrameSource
    source = FrameSource(args.input,args)
    try:
        frame = source.seek(0)
        expected_pixels = source.metadata.get('validation', {}).get('decoded_yuv_sha256')
        if expected_pixels and hashlib.sha256(frame.raw).hexdigest() != expected_pixels:
            raise ValueError('Decoded image differs from recorded validation')
        if not verify_only:
            target = Path(args.output_file)
            target.parent.mkdir(parents=True,exist_ok=True)
            rgb = frame.rgb[:source.metadata['image_height'],:source.metadata['image_width']]
            with tempfile.TemporaryDirectory(prefix='.uf-png-',dir=target.parent) as temporary:
                stage = Path(temporary)/'image.png'
                Image.fromarray(rgb).save(stage)
                with stage.open('rb') as stream: os.fsync(stream.fileno())
                os.link(stage,target); sync_directory(target.parent)
        event('verification_completed' if verify_only else 'decode_completed',archive=str(args.input),frames=1,runtime=source.metadata['pipeline'].get('runtime','fp16'))
        return 0
    finally:
        source.close()


def cleanup_image(args):
    target = Path(args.input)
    pool = ImagePool(instance_name='cleanup')
    metadata,_ = read_image(target)
    pool.ensure_pipeline(target.parent,metadata['pipeline'])
    lease,_ = pool.try_acquire(target.parent,target.name)
    if lease is None:
        raise RuntimeError('Thumbnail owned by another instance')
    try:
        with Heartbeat(lease) as pulse:
            metadata,_ = read_image(target)
            source = Path(metadata['source']['path'])
            if getattr(args,'base_root',None):
                base = Path(args.base_root).resolve()
                source = base / metadata['source']['relative_path']
                if not source.resolve().is_relative_to(base):
                    raise ValueError('Original path escapes base_root')
            if not source.exists(): return 0
            before = identity(source)
            if source.is_symlink() or sha256(source)!=metadata['source']['sha256']:
                raise ValueError('Original thumbnail changed; retaining it')
            if not args.yes:
                event('cleanup_preview',source=str(source),bytes=source.stat().st_size); return 0
            decode_image(args,verify_only=True)
            if identity(source)!=before or sha256(source)!=metadata['source']['sha256']:
                raise ValueError('Original thumbnail changed during verification')
            pulse.check(); source.unlink(); sync_directory(source.parent)
            event('original_deleted',source=str(source),archive=str(target))
            return 0
    finally:
        lease.release()


def _pass_entry(writer,jobs,config,paths,device,instance,events,source_base):
    from src.archive.dashboard import install
    install(events)
    try:
        writer.send(encode_images(jobs,config,paths,device,instance,source_base=source_base))
    finally:
        writer.close()


def run_pass(args,config,paths):
    if getattr(args,'thumbnail_codec','keep') != 'dcvc-intra': return set()
    import multiprocessing
    from src.archive.dashboard import event_queue
    jobs = args._thumbnail_jobs if hasattr(args,"_thumbnail_jobs") else discover(args)
    if not jobs: return set()
    args._cleanup_sources.update(str(p.absolute()) for p,_ in jobs)
    config = {**config,'qp_i':args.thumbnail_qp,'qp_p':args.thumbnail_qp}
    context = multiprocessing.get_context('spawn')
    parent,child = context.Pipe(duplex=False)
    process = context.Process(target=_pass_entry,args=(child,jobs,config,paths,
          'cpu' if args.device=='cpu' else args.cuda_idx[0],args.shared_instance,event_queue(),Path(args.base_root).resolve() if args.base_root else Path(args.input_file).absolute().parent))
    try:
        process.start(); child.close()
        while process.is_alive() and not parent.poll(1): pass
        failed = parent.recv() if parent.poll() else {str(p.parent) for _,p in jobs}
        process.join()
        if process.exitcode: failed.update(str(p.parent) for _,p in jobs)
        return failed
    finally:
        if process.is_alive(): process.terminate(); process.join()
        parent.close(); child.close()
