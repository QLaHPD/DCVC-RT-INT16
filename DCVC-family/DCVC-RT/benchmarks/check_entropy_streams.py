"""Check sparse escape values, mixed Y/Z tasks, and one/two-coder transitions.

Save reference streams with the old C++ extension, then compare with the new one.
Each task includes ample compressible padding for the original coder's buffer.
"""
import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--extension', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--save', type=Path)
    mode.add_argument('--compare', type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('MLCodec_extensions_cpp', args.extension)
    ext = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ext
    spec.loader.exec_module(ext)
    encoder = ext.RansEncoder()
    decoder = ext.RansDecoder()
    # Normal values 0/1 plus escape; offsets span every signed-byte value.
    cdfs = np.tile(np.array([0, 65488, 65520, 65536], dtype=np.int32), (256, 1))
    sizes = np.full(256, 4, dtype=np.int32)
    offsets = np.arange(-128, 128, dtype=np.int32)
    group = encoder.add_cdf(cdfs, sizes, offsets)
    assert decoder.add_cdf(cdfs, sizes, offsets) == group
    streams = {}
    expected = np.load(args.compare, allow_pickle=False) if args.compare else None
    for iteration, two in enumerate((False, True, False, True)):
        encoder.set_use_two_encoders(two)
        decoder.set_use_two_decoders(two)
        for offset in (0, 63, 127):
            encoder.reset()
            # Z's channel split stays aligned in two-coder mode.
            z = np.repeat(offsets[offset:offset+2].astype(np.int8), 4096)
            z[::32] = np.resize(np.arange(-128, 128, dtype=np.int16).astype(np.int8), z[::32].size)
            indexes = np.tile(np.arange(256, dtype=np.uint8), 32)
            symbols = offsets[indexes].astype(np.int16)
            # Every possible signed symbol is exercised against both signs of
            # offset, including raw_val=0 and multiple bypass count groups.
            symbols[:256] = np.arange(-128, 128, dtype=np.int16)
            indexes[:256] = offset
            packed = ((symbols.astype(np.int32) << 8) | indexes).astype(np.int16)
            encoder.encode_z(z, group, offset, 4096)
            encoder.encode_y(packed, group)
            encoder.encode_y(packed[::-1].copy(), group)
            encoder.flush()
            encoded = encoder.get_encoded_stream()
            key = f'{iteration}_{two}_{offset}'
            streams[key] = encoded.copy()
            if expected is not None:
                assert np.array_equal(encoded, expected[key]), key
            decoder.set_stream(encoded)
            decoder.decode_z(z.size, group, offset, 4096)
            assert np.array_equal(decoder.get_decoded_tensor(), z), key
            assert np.array_equal(decoder.decode_and_get_y(indexes, group), symbols.astype(np.int8)), key
            assert np.array_equal(decoder.decode_and_get_y(indexes[::-1].copy(), group), symbols[::-1].astype(np.int8)), key
    if args.save:
        np.savez(args.save, **streams)
    print(f'PASS: {len(streams)} entropy streams and decoded symbols, mixed tasks and coder transitions')


if __name__ == '__main__':
    main()
