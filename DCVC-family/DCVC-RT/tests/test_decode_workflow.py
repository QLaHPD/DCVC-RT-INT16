from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.cli.decode_workflow import build_decode_tasks, configure_parser


class DecodeInputTests(unittest.TestCase):
    def test_specific_bin_builds_one_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bitstream = root / "selected_176x96_qI35_qP14.bin"
            bitstream.write_bytes(b"fixture")
            args = SimpleNamespace(
                input_file=str(bitstream),
                input_folder=None,
                output_folder=str(root / "decoded"),
                original_folder=None,
            )

            tasks, total, already_done = build_decode_tasks(args)

            self.assertEqual(total, 1)
            self.assertEqual(already_done, 0)
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0].bin_path, str(bitstream))
            self.assertEqual(tasks[0].base_name, bitstream.stem)

    def test_specific_input_must_be_a_bin_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wrong_type = root / "selected.txt"
            wrong_type.write_text("fixture", encoding="utf-8")
            args = SimpleNamespace(
                input_file=str(wrong_type),
                input_folder=None,
                output_folder=str(root / "decoded"),
                original_folder=None,
            )

            tasks, total, already_done = build_decode_tasks(args)

            self.assertEqual((tasks, total, already_done), ([], 0, 0))

    def test_specific_intra_image_builds_image_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bitstream = root / "selected.webp_qI45.dcvci"
            bitstream.write_bytes(b"fixture")
            args = SimpleNamespace(
                input_file=str(bitstream),
                input_folder=None,
                output_folder=str(root / "decoded"),
                original_folder=None,
            )

            tasks, total, already_done = build_decode_tasks(args)

            self.assertEqual((total, already_done, len(tasks)), (1, 0, 1))
            self.assertTrue(tasks[0].is_intra_image)

    def test_image_filter_ignores_video_bitstreams(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "video.bin").write_bytes(b"video")
            image = root / "thumbnail.webp_qI30.dcvci"
            image.write_bytes(b"image")
            args = SimpleNamespace(
                input_file=None,
                input_folder=str(root),
                input_kind="images",
                output_folder=str(root / "decoded"),
                original_folder=None,
            )

            tasks, total, already_done = build_decode_tasks(args)

            self.assertEqual((total, already_done), (1, 0))
            self.assertEqual([Path(task.bin_path) for task in tasks], [image])

    def test_video_filter_ignores_intra_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.bin"
            video.write_bytes(b"video")
            (root / "thumbnail.webp_qI30.dcvci").write_bytes(b"image")
            args = SimpleNamespace(
                input_file=None,
                input_folder=str(root),
                input_kind="videos",
                output_folder=str(root / "decoded"),
                original_folder=None,
            )

            tasks, total, already_done = build_decode_tasks(args)

            self.assertEqual((total, already_done), (1, 0))
            self.assertEqual([Path(task.bin_path) for task in tasks], [video])

    def test_file_and_folder_arguments_are_mutually_exclusive(self):
        parser = argparse.ArgumentParser()
        configure_parser(parser)
        with self.assertRaises(SystemExit):
            parser.parse_args([
                "--input_file", "one.bin",
                "--input_folder", "inputs",
                "--output_folder", "decoded",
            ])


if __name__ == "__main__":
    unittest.main()
