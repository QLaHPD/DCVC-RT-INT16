"""Malformed entropy buffers must raise Python errors and leave workers usable."""
import unittest

import numpy as np
from MLCodec_extensions_cpp import RansDecoder, RansEncoder


class EntropyBoundsTests(unittest.TestCase):
    def pair(self, parallel=1):
        cdf = np.array([[0, 16384, 32768, 49152, 65536]], dtype=np.int32)
        sizes = np.array([5], dtype=np.int32)
        encoder, decoder = RansEncoder(), RansDecoder()
        for obj in (encoder, decoder):
            obj.set_entropy_coder_parallel(parallel)
            for kind in (0, 1):
                obj.set_cdf(cdf, sizes, kind)
        return encoder, decoder

    def test_short_headers_rejected_for_every_parallelism(self):
        for parallel in range(1, 9):
            _, decoder = self.pair(parallel)
            for length in range(4):
                with self.subTest(parallel=parallel, length=length), self.assertRaises(RuntimeError):
                    decoder.set_stream(np.zeros(length, dtype=np.uint8))

    def test_invalid_parallel_offsets_are_rejected(self):
        for parallel in range(3, 9):
            _, decoder = self.pair(parallel)
            for offset in (0, 3, 0x7fffffff, 0xffffffff):
                stream = np.frombuffer(offset.to_bytes(4, 'little') + bytes(32), dtype=np.uint8)
                with self.subTest(parallel=parallel, offset=offset), self.assertRaises(RuntimeError):
                    decoder.set_stream(stream)

    def test_exhausted_payload_raises_and_decoder_recovers(self):
        for parallel in range(1, 9):
            with self.subTest(parallel=parallel):
                encoder, decoder = self.pair(parallel)
                symbols = np.resize(np.array([-128, -1, 0, 1, 127], dtype=np.int8), 1000)
                packed = symbols.astype(np.int16) << 8  # CDF index zero in low byte.
                encoder.reset()
                encoder.encode_y(packed)
                encoder.flush()
                stream = encoder.get_encoded_stream()
                indexes = np.zeros(symbols.size, dtype=np.uint8)
                decoder.set_stream(stream)
                decoder.decode_y(np.zeros(100000, dtype=np.uint8))
                with self.assertRaises(RuntimeError):
                    decoder.get_decoded_tensor()
                decoder.set_stream(stream)
                decoder.decode_y(indexes)
                np.testing.assert_array_equal(decoder.get_decoded_tensor(), symbols)

    def test_invalid_initial_state_and_cdf_are_rejected(self):
        _, decoder = self.pair()
        with self.assertRaises(RuntimeError):
            decoder.set_stream(np.zeros(4, dtype=np.uint8))
        for cdf in ([[0, 0, 65536]], [[0, 16384, 65535]], [[1, 16384, 65536]]):
            with self.assertRaises(ValueError):
                decoder.set_cdf(np.array(cdf, dtype=np.int32), np.array([3], dtype=np.int32), 1)


if __name__ == '__main__':
    unittest.main()
