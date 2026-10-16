"""Lazy entropy workers preserve the established one-to-eight-stream format."""
import gc
import hashlib
from pathlib import Path
import unittest

import numpy as np

from MLCodec_extensions_cpp import RansEncoder, RansDecoder


GOLDENS = (
    'e993a970a11765e45e49486525436c584f57697ba1c10a21d9229817ac1a41f6',
    '3d792a9f80c8567661eceee2ada7888f976382e5bd1ffabe85f6e1e120a373eb',
    'cbbfa4c91525d7d5e2986429886538c352d414435a672da2e1c76833168b897f',
    'd9d3d634e02c2e590aea8adaa13a52e181e2960ab06e7e098bebb7c460a9d966',
    '7621492e96a2cdea5570fc81017bf89fb8aa6589d47b5e6a5c4cc7318003694c',
    'dcfc48b8ab78d42a7c4110da525027df5ec3626eabdef264e75a47f32166a591',
    '48e61585db1547f977f16fb5665b074982741a0657adf711372aa12e354dbb89',
    '072a8e591a6d1da735e7ea98030f4d698753fa84ff9f8b54bdd458b345fe1b10',
)


class EntropyWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not callable(getattr(RansDecoder, 'decode_y_with_skip', None)):
            raise unittest.SkipTest('Rebuild UF entropy extension for lazy masked decoding')

    def test_parallel_growth_shrink_and_legacy_bytes(self):
        cdf = np.tile(np.linspace(0, 65536, 18, dtype=np.int32), (3, 1))
        lengths = np.full(3, 18, dtype=np.int32)
        symbols = [np.array([-128, -127, -9, -8, -1, 0, 1, 7, 8, 9, 126, 127]*23, dtype=np.int8),
                   np.array([127, 0, -128, 3, -3, 1, -1], dtype=np.int8)]
        indexes = [(np.arange(s.size) % 3).astype(np.uint8) for s in symbols]
        z = np.array([-128, -9, -1, 0, 1, 9, 127]*5, dtype=np.int8)
        encoder, decoder = RansEncoder(), RansDecoder()
        # CDFs are installed before growing the pool, as on a reused codec.
        for obj in (encoder, decoder):
            for kind in (0, 1):
                obj.set_cdf(cdf, lengths, kind)
        for parallel in (1, 3, 2, 8, 4, 7, 5, 6, 1):
            with self.subTest(parallel=parallel):
                for obj in (encoder, decoder):
                    obj.set_entropy_coder_parallel(parallel)
                encoder.reset()
                for values, index in reversed(list(zip(symbols, indexes))):
                    encoder.encode_y((values.astype(np.int16) << 8) | index.astype(np.int16))
                encoder.encode_z(z, 1, 2)
                encoder.flush()
                bits = encoder.get_encoded_stream()
                self.assertEqual(hashlib.sha256(bits.tobytes()).hexdigest(), GOLDENS[parallel-1])
                decoder.set_stream(bits)
                decoder.decode_z(z.size, 1, 2)
                self.assertTrue(np.array_equal(decoder.get_decoded_tensor(), z))
                for values, index in zip(symbols, indexes):
                    decoder.decode_y(index)
                    self.assertTrue(np.array_equal(decoder.get_decoded_tensor(), values))

    def test_invalid_parallelism_is_rejected(self):
        for obj in (RansEncoder(), RansDecoder()):
            for parallel in (-1, 0, 9):
                with self.subTest(parallel=parallel):
                    with self.assertRaises(ValueError):
                        obj.set_entropy_coder_parallel(parallel)

    def test_masked_decoder_skips_without_advancing_and_retains_results(self):
        cdf = np.tile(np.linspace(0, 65536, 18, dtype=np.int32), (3, 1))
        lengths = np.full(3, 18, dtype=np.int32)
        encoder, decoder = RansEncoder(), RansDecoder()
        for obj in (encoder, decoder):
            obj.set_cdf(cdf, lengths, 1)
        values = np.arange(-128, 128, dtype=np.int8)
        mixed = (np.arange(256) % 3).astype(np.int16)
        mixed[::3] = -1
        parts = [mixed, np.full(19, -1, dtype=np.int16), np.empty(0, dtype=np.int16),
                 (np.arange(256) % 3).astype(np.int16)]
        encoder.reset()
        expected = []
        for indexes in parts:
            result = np.zeros(indexes.size, dtype=np.int8)
            result[indexes >= 0] = values[:indexes.size][indexes >= 0]
            expected.append(result)
        for indexes, result in reversed(list(zip(parts, expected))):
            active = indexes >= 0
            if active.any():
                encoder.encode_y((result[active].astype(np.int16) << 8) | indexes[active])
        encoder.flush()
        bits = encoder.get_encoded_stream()
        for noncontiguous in (False, True):
            decoder.set_stream(bits)
            results = []
            for indexes in parts:
                if noncontiguous:
                    storage = np.zeros(indexes.size*2, dtype=np.int16)
                    storage[::2] = indexes
                    indexes = storage[::2]
                results.append(decoder.decode_y_with_skip(indexes))
            for actual, reference in zip(results, expected):
                self.assertTrue(np.array_equal(actual, reference))

    def test_masked_decoder_rejects_parallel_and_unknown_tables(self):
        decoder = RansDecoder()
        for indexes in (np.array([0], dtype=np.int16), np.array([256], dtype=np.int16)):
            with self.assertRaises(ValueError):
                decoder.decode_y_with_skip(indexes)
        self.assertEqual(decoder.decode_y_with_skip(np.array([-1, -9], dtype=np.int16)).tolist(), [0, 0])
        decoder.set_entropy_coder_parallel(2)
        with self.assertRaises(ValueError):
            decoder.decode_y_with_skip(np.array([-1], dtype=np.int16))

    @unittest.skipUnless(Path('/proc/self/task').is_dir(), 'Linux thread accounting required')
    def test_default_coder_starts_only_two_workers(self):
        gc.collect()
        before = len(list(Path('/proc/self/task').iterdir()))
        encoder, decoder = RansEncoder(), RansDecoder()
        after = len(list(Path('/proc/self/task').iterdir()))
        self.assertEqual(after-before, 2)
        del encoder, decoder
        gc.collect()
        self.assertEqual(len(list(Path('/proc/self/task').iterdir())), before)


if __name__ == '__main__':
    unittest.main()
