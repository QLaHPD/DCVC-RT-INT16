#!/usr/bin/env python3
"""Cross-decode local synthetic UF streams using saved and current Python models."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Local synthetic video')
    parser.add_argument('--reference-model', required=True, help='Saved src/int16/model.py from before changes')
    parser.add_argument('--reference-layers', help='Saved src/int16/layers.py from before changes')
    parser.add_argument('--reference-native', help='Saved uf_int16_cuda extension from before changes')
    parser.add_argument('--output', required=True, help='New output directory')
    parser.add_argument('--variant', choices=('htl', 'hts', 'ld'), default='htl')
    args = parser.parse_args()
    from src.archive.media import FrameReader, dimensions, probe
    from src.int16.codec import IntegerCodec
    from src.int16.model import IntegerModel
    from src.int16.layers import Layers
    import uf_int16_cuda

    current_native = old_native = uf_int16_cuda
    if args.reference_native:
        spec = importlib.util.spec_from_file_location('uf_int16_reference.uf_int16_cuda', args.reference_native)
        old_native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old_native)

    old_layers = Layers
    if args.reference_layers:
        spec = importlib.util.spec_from_file_location('src.int16._compat_layers', args.reference_layers)
        layer_reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(layer_reference)
        old_layers = layer_reference.Layers

    spec = importlib.util.spec_from_file_location('src.int16._compat_reference', args.reference_model)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    codec = IntegerCodec(ROOT / 'checkpoints/cvpr2026_image.pth.tar',
                         ROOT / f'checkpoints/cvpr2026_video_{args.variant}.pth.tar', args.variant,
                         prepared_i=next((ROOT / 'checkpoints').glob('cvpr2026_image.*.int16.pt')),
                         prepared_p=next((ROOT / 'checkpoints').glob(f'cvpr2026_video_{args.variant}.*.int16.pt')))
    assert codec.video._prior_native is not None, 'fused extension is missing'

    def select(implementation):
        previous = implementation is reference.IntegerModel
        native = old_native if previous else current_native
        sys.modules['uf_int16_cuda'] = native
        for model in (codec.image, codec.video):
            # Switch the whole Python implementation, including helpers called
            # by compress/decompress, rather than only their outer wrappers.
            model.__class__ = implementation
            model.layers.__class__ = old_layers if previous else Layers
            model._prior_native = native if all(hasattr(native, n) for n in
                ('prior_encode', 'prior_indexes', 'prior_restore')) else None
            # This callable is cached when weights are loaded. Swap it too so
            # every reference block uses the saved native library.
            if hasattr(model.layers, '_native_block'):
                model.layers._native_block = getattr(native, 'depth_conv_block', None)
            if hasattr(model.layers, '_native_packed_block'):
                model.layers._native_packed_block = getattr(native, 'depth_conv_block_packed', None)
    source = Path(args.input).resolve()
    media = probe(source)
    cases = [dict(skip=0, reset=0, intra=-1, frames=17, resolution=64, qi=36, qp=30),
             dict(skip=0.2, reset=2, intra=-1, frames=20, resolution=64, qi=10, qp=14),
             dict(skip=16, reset=-1, intra=8, frames=19, resolution=144, qi=63, qp=63),
             dict(skip=0.2, reset=2, intra=1, frames=3, resolution=62, qi=0, qp=0)]
    records = []
    for index, case in enumerate(cases):
        width, height = dimensions(media['width'], media['height'], case['resolution'])
        codec.set_skip_threshold(case['skip'])
        paths = {}
        for mode, implementation in (('reference', reference.IntegerModel), ('current', IntegerModel)):
            select(implementation)
            path = output / f'{index}-{mode}.bin'
            with FrameReader(source, width, height, media, decoder_threads=1, prefetch_frames=8) as reader:
                result = codec.encode(reader, path, case['qi'], case['qp'], case['reset'],
                                      case['intra'], case['frames'])
            paths[mode] = path
        assert paths['reference'].read_bytes() == paths['current'].read_bytes(), f'changed bitstream: {case}'
        metadata = dict(width=width, height=height, frames=result['frames'], packets=result['packets'])
        validations = []
        for implementation, mode in ((IntegerModel, 'reference'), (reference.IntegerModel, 'current')):
            select(implementation)
            validations.append(codec.decode(paths[mode], metadata))
        assert validations[0]['decoded_yuv_sha256'] == validations[1]['decoded_yuv_sha256'], f'changed pixels: {case}'
        record = dict(variant=args.variant, case=case, video=metadata,
                      current_native_sha256=hashlib.sha256(Path(current_native.__file__).read_bytes()).hexdigest(),
                      reference_native_sha256=hashlib.sha256(Path(old_native.__file__).read_bytes()).hexdigest(),
                      bitstream_sha256=hashlib.sha256(paths['current'].read_bytes()).hexdigest(),
                      decoded_yuv_sha256=validations[0]['decoded_yuv_sha256'],
                      current_decodes_reference=True, reference_decodes_current=True)
        records.append(record)
        print(json.dumps(record), flush=True)
        (output / 'results.json').write_text(json.dumps(records, indent=2)+'\n')
    select(IntegerModel)


if __name__ == '__main__':
    main()
