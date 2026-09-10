from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.cli.progress import append_progress_log
from src.cli.thumbnail_codec import (
    THUMBNAIL_PROGRESS_FILENAME,
    discover_thumbnails,
    inspect_intra_image_bitstream,
    thumbnail_output_name,
)
from src.utils.stream_helper import write_ip, write_sps


def write_image(path: Path, size=(12, 7)):
    Image.new("RGB", size, (30, 120, 210)).save(path)


def write_intra(path: Path, width=12, height=7, qp=45, i_frame=True, frames=1):
    with path.open("wb") as stream:
        write_sps(stream, {
            "sps_id": 0,
            "height": height,
            "width": width,
            "ec_part": 0,
            "use_ada_i": 0,
        })
        for _ in range(frames):
            write_ip(stream, i_frame, 0, qp, b"synthetic-entropy")


class ThumbnailCodecTests(unittest.TestCase):
    def test_standalone_intra_stream_requires_exactly_one_i_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid.dcvci"
            write_intra(valid)
            info = inspect_intra_image_bitstream(valid)
            self.assertEqual((info.width, info.height, info.qp), (12, 7, 45))

            predictive = root / "predictive.dcvci"
            write_intra(predictive, i_frame=False)
            with self.assertRaisesRegex(ValueError, "I-frame"):
                inspect_intra_image_bitstream(predictive)

            multiple = root / "multiple.dcvci"
            write_intra(multiple, frames=2)
            with self.assertRaisesRegex(ValueError, "exactly one"):
                inspect_intra_image_bitstream(multiple)

    def test_discovery_resumes_only_matching_source_identity_and_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            source = input_dir / "thumb.webp"
            write_image(source)

            first = discover_thumbnails("CHANNEL", input_dir, output_dir, qp=45)
            self.assertEqual(first.total_found, 1)
            self.assertEqual(len(first.pending_tasks), 1)
            task = first.pending_tasks[0]
            output = output_dir / thumbnail_output_name(source, 45)
            write_intra(output)
            append_progress_log(output_dir, {
                "source_relative": task.source_relative,
                "status": "done",
                "source_path": task.source_path,
                "source_size": task.source_size,
                "source_mtime_ns": task.source_mtime_ns,
                "source_device": task.source_device,
                "source_inode": task.source_inode,
                "width": task.width,
                "height": task.height,
                "qp": task.qp,
                "output": task.output_path,
            }, THUMBNAIL_PROGRESS_FILENAME)

            resumed = discover_thumbnails("CHANNEL", input_dir, output_dir, qp=45)
            self.assertEqual(resumed.already_done, 1)
            self.assertEqual(resumed.pending_tasks, [])

            write_image(source, size=(13, 7))
            changed = discover_thumbnails("CHANNEL", input_dir, output_dir, qp=45)
            self.assertEqual(changed.already_done, 0)
            self.assertEqual(len(changed.pending_tasks), 1)

    def test_recursive_duplicate_names_are_rejected_before_encoding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            output_dir = root / "output"
            (input_dir / "a").mkdir(parents=True)
            (input_dir / "b").mkdir(parents=True)
            write_image(input_dir / "a" / "same.png")
            write_image(input_dir / "b" / "same.png")

            discovery = discover_thumbnails(
                "CHANNEL", input_dir, output_dir, qp=45, recursive=True
            )
            self.assertEqual(discovery.total_found, 2)
            self.assertTrue(any("collision" in error for error in discovery.errors))


if __name__ == "__main__":
    unittest.main()
