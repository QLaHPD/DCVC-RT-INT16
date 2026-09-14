"""Random access UF decoding behind the shared RT viewer's frame-source API."""
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from src.utils.stream_helper import NalType, read_header, read_ip_remaining, read_sps_remaining


@dataclass
class Packet:
    offset: int
    first: int
    count: int
    nal_type: NalType
    sps: dict
    qp: int


def packet_index(stream, metadata, chunk_size):
    table, packets, count = {}, [], 0
    while stream.peek(1):
        offset = stream.tell()
        header = read_header(stream)
        if header['nal_type'] == NalType.NAL_SPS:
            sps = read_sps_remaining(stream, header['sps_id'])
            if (sps['width'], sps['height']) != (metadata['width'], metadata['height']):
                raise ValueError('Bitstream dimensions differ from metadata')
            table[header['sps_id']] = sps
            continue
        if header['nal_type'] not in (NalType.NAL_I, NalType.NAL_P):
            raise ValueError('Unsupported UF packet')
        if not packets and header['nal_type'] != NalType.NAL_I:
            raise ValueError('UF stream must start with an I-frame')
        qp, _, _, _ = read_ip_remaining(stream)
        length = min(1 if header['nal_type'] == NalType.NAL_I else chunk_size, metadata['frames'] - count)
        if length <= 0:
            raise ValueError('Extra UF packet')
        packets.append(Packet(offset, count, length, header['nal_type'], table[header['sps_id']].copy(), qp))
        count += length
    if count != metadata['frames'] or len(packets) != metadata['packets']:
        raise ValueError('Incomplete UF bitstream')
    return packets


def raw_rgb(raw, width, height):
    """Display conversion only; entropy decoding stays in its recorded arithmetic."""
    area = width * height
    data = np.frombuffer(raw, dtype=np.uint8)
    y = data[:area].reshape(height, width).astype(np.float32)
    uv = data[area:].reshape(2, height//2, width//2).repeat(2, 1).repeat(2, 2).astype(np.float32) - 128
    # Match the full-range BT.709 YUV transform used by UF.
    u, v = uv
    return np.stack((y + 1.5748*v, y - .187324*u - .468124*v, y + 1.8556*u), -1).clip(0, 255).round().astype(np.uint8)


class FrameSource:
    def __init__(self, path, options):
        from src.archive.storage import load_bundle, artifact_path
        self.path, self.options = Path(path), options
        self.image = self.path.suffix.lower() == '.dcvci'
        if self.image:
            from src.archive.thumbnails import read_image
            self.metadata, self.payload_offset = read_image(path)
            self.bitstream = self.path
        else:
            root, self.metadata = load_bundle(path)
            self.bitstream = artifact_path(root, self.metadata, 'video.bin')
            self.payload_offset = 0
        self.codec = None
        self.stream = open(self.bitstream, 'rb')
        try:
            self.stream.seek(self.payload_offset)
            # Validate the preamble without loading networks on the UI thread.
            config = self.metadata['pipeline']
            if config.get('runtime') == 'int16':
                from src.int16.codec import MAGIC
                ids = config['integer']
                expected = MAGIC + bytes.fromhex(ids['image']) + (bytes(32) if self.image else bytes.fromhex(ids['video']))
                if self.stream.read(len(expected)) != expected:
                    raise ValueError('UF runtime/model identity mismatch')
            elif self.stream.peek(4)[:4] == b'UF16':
                raise ValueError('INT16 payload has float metadata')
            self.packets = packet_index(self.stream, self.metadata['video'], 1 if config['variant'] == 'ld' else 8)
            self.index = SimpleNamespace(path=self.path, frame_count=self.metadata['video']['frames'], frames=self.packets)
            self.starts = [p.first for p in self.packets]
            self.anchors = [i for i,p in enumerate(self.packets) if p.nal_type == NalType.NAL_I]
            self.next_packet, self.cached_packet, self.cached = 0, -1, []
        except BaseException:
            self.stream.close()
            raise

    def seek(self, frame_index, output_format='rgb'):
        import torch
        from src.archive.workflow import checked_codec
        if not 0 <= frame_index < self.index.frame_count:
            raise IndexError(frame_index)
        if self.codec is None:
            if self.image:
                key = repr(self.metadata['pipeline'])
                cache = getattr(self.options,'_uf_image_decoder',None)
                if cache and cache[0] == key:
                    self.codec = cache[1]
                else:
                    self.options._uf_image_decoder = None
                    self.codec = checked_codec(self.metadata, self.options)
                    self.options._uf_image_decoder = (key,self.codec)
            else:
                self.codec = checked_codec(self.metadata, self.options)
        if self.codec.device.type == 'cuda':
            torch.cuda.set_device(self.codec.device)
            torch.cuda.set_stream(self.codec.stream)
        target = bisect_right(self.starts, frame_index) - 1
        if target != self.cached_packet:
            anchor = self.anchors[bisect_right(self.anchors, target) - 1]
            if target < self.next_packet or self.next_packet < anchor:
                self.next_packet = anchor
            with torch.no_grad():
                while self.next_packet <= target:
                    packet = self.packets[self.next_packet]
                    self.stream.seek(packet.offset)
                    read_header(self.stream)
                    qp, ec, reset, bits = read_ip_remaining(self.stream)
                    if packet.nal_type == NalType.NAL_I:
                        result = self.codec.image.decompress(bits, packet.sps, qp, ec)
                        if self.codec.video is not None:
                            self.codec.video.clear_dpb()
                            self.codec.video.add_ref_feature_from_frame(result['x_hat'], apply_feature_adaptor=False)
                    else:
                        result = self.codec.video.decompress(bits, packet.sps, qp, ec, reset)
                    frames = result['x_hat'] if isinstance(result['x_hat'], list) else [result['x_hat']]
                    expected = 1 if packet.nal_type == NalType.NAL_I else (1 if self.metadata['pipeline']['variant']=='ld' else 8)
                    if len(frames) != expected:
                        raise ValueError('Unexpected UF decoder chunk length')
                    if any(torch.is_tensor(f) and not torch.isfinite(f).all() for f in frames):
                        raise ValueError('Decoder produced non-finite pixels')
                    self.cached = [self.codec.raw_frame(f, packet.sps['width'], packet.sps['height']) for f in frames[:packet.count]]
                    self.cached_packet = self.next_packet
                    self.next_packet += 1
        packet = self.packets[target]
        width, height = packet.sps['width'], packet.sps['height']
        raw = self.cached[frame_index-packet.first]
        rgb = raw_rgb(raw,width,height)
        if self.image:
            width,height = self.metadata['image_width'],self.metadata['image_height']
            rgb = rgb[:height,:width]
        return SimpleNamespace(index=frame_index, width=width, height=height,
                               frame_type='I' if packet.nal_type == NalType.NAL_I else 'P', qp=packet.qp,
                               rgb=rgb, raw=raw)

    def close(self):
        self.stream.close()
        self.codec = None
        self.cached = []


def run_view(args):
    import tkinter as tk
    from src.archive.rt_managed import managed_module
    viewer = managed_module('viewer')
    path = Path(args.input)
    images = viewer.discover_intra_images(path, args.recursive) if path.is_dir() and not (path/'manifest.json').exists() else ([path] if path.suffix.lower()=='.dcvci' else [])
    if path.is_dir() and not images and not (path/'manifest.json').exists():
        raise ValueError('No .dcvci images found')
    root = tk.Tk()
    root.withdraw()
    if images:
        viewer.ImageGalleryApp(root, images, args, args.start_frame, args.max_width,args.max_height,
                               'UF / recorded runtime', source_factory=FrameSource)
        root.title('DCVC-UF Intra Image Viewer')
    else:
        source = FrameSource(path,args)
        viewer.ViewerApp(root,source,args.fps or source.metadata['video']['fps'],args.start_frame,
                         args.max_width,args.max_height,source.metadata['pipeline'].get('runtime','fp16'))
        root.title(f'DCVC-UF Viewer - {path.name}')
    root.deiconify()
    root.mainloop()
    return 0
