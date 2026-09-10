from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.codec.frame_decoder import configure_decode_runtime
from src.cli.viewer import discover_intra_images


class DecodeRuntimeTests(unittest.TestCase):
    def test_explicit_int16_enables_verified_runtime(self):
        with mock.patch.dict(os.environ, {"DCVC_USE_INT16": "0"}), \
                mock.patch("src.codec.frame_decoder.torch.cuda.is_available", return_value=True), \
                mock.patch(
                    "src.codec.frame_decoder.int16_runtime.CUSTOMIZED_INT16_CUDA_INFERENCE",
                    True,
                ):
            self.assertEqual(configure_decode_runtime("int16", True), "CUDA INT16")
            self.assertEqual(os.environ["DCVC_USE_INT16"], "1")

    def test_explicit_int16_refuses_silent_float_fallback(self):
        with mock.patch.dict(os.environ, {"DCVC_USE_INT16": "0"}), \
                mock.patch("src.codec.frame_decoder.torch.cuda.is_available", return_value=True), \
                mock.patch(
                    "src.codec.frame_decoder.int16_runtime.CUSTOMIZED_INT16_CUDA_INFERENCE",
                    False,
                ):
            with self.assertRaisesRegex(ValueError, "INT16 CUDA extension is unavailable"):
                configure_decode_runtime("int16", True)
            self.assertEqual(os.environ["DCVC_USE_INT16"], "0")

    def test_explicit_float_disables_int16(self):
        with mock.patch.dict(os.environ, {"DCVC_USE_INT16": "1"}), \
                mock.patch("src.codec.frame_decoder.torch.cuda.is_available", return_value=False):
            self.assertEqual(configure_decode_runtime("float", False), "CPU float")
            self.assertEqual(os.environ["DCVC_USE_INT16"], "0")


class ImageDiscoveryTests(unittest.TestCase):
    def test_gallery_selects_only_dcvci_and_honors_recursive_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.dcvci"
            first.write_bytes(b"fixture")
            (root / "video.bin").write_bytes(b"fixture")
            nested = root / "nested"
            nested.mkdir()
            second = nested / "second.DCVCI"
            second.write_bytes(b"fixture")

            self.assertEqual(discover_intra_images(root), [first])
            self.assertEqual(discover_intra_images(root, recursive=True), [first, second])


if __name__ == "__main__":
    unittest.main()
