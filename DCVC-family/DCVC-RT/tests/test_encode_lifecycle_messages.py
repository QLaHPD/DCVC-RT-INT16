from __future__ import annotations

import queue
import argparse
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from src.cli.channel_lifecycle import validate_channel, write_channel_inventory
from src.cli.encode_workflow import (
    EncoderCfg,
    EncodeTask,
    append_progress_log,
    configure_parser,
    process_one_file,
    run,
    write_bytes_final,
)
from src.utils.stream_helper import write_ip, write_sps


def write_synthetic_av_fixture(path: Path):
    subprocess.run([
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=176x96:rate=24",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
        "-t", "1", "-c:v", "ffv1", "-c:a", "pcm_s16le", str(path),
    ], check=True)


class FakeNeuralEncoder:
    """Drains real ffmpeg frames and emits a structurally valid test bitstream."""

    def encode_from_ffmpeg_rawpipe(
        self, ffmpeg_proc, width, height, out_path, log_dir, video_id,
        on_progress=None, finalize_mode="atomic",
    ):
        raw = ffmpeg_proc.stdout.read()
        frame_size = width * height * 3 // 2
        frames = len(raw) // frame_size
        if frames <= 0 or len(raw) % frame_size:
            raise RuntimeError("fake encoder did not receive complete YUV420 frames")
        from io import BytesIO
        stream = BytesIO()
        write_sps(stream, {
            "sps_id": 0,
            "height": height,
            "width": width,
            "ec_part": 0,
            "use_ada_i": 0,
        })
        for index in range(frames):
            write_ip(stream, index == 0, 0, 14, b"fake-entropy")
        append_progress_log(log_dir, {
            "video_id": video_id,
            "stage": "stats",
            "event": "frames_encoded",
            "frames": frames,
        })
        write_bytes_final(out_path, stream.getbuffer(), mode=finalize_mode)
        if on_progress:
            on_progress(frames, float(frames), 1.0)


class EncodeLifecycleMessageTests(unittest.TestCase):
    def test_more_than_three_workers_is_rejected_before_startup(self):
        parser = argparse.ArgumentParser()
        configure_parser(parser)
        args = parser.parse_args([
            "--base_root", "/does/not/matter",
            "--output_root", "/does/not/matter",
            "--channel_ids", "TEST_CHANNEL",
            "--procs", "4",
        ])
        self.assertEqual(run(args), 2)

    def test_real_ffmpeg_audio_and_fake_video_reach_cleanup_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input" / "TEST_CHANNEL"
            output_dir = root / "output" / "TEST_CHANNEL"
            input_dir.mkdir(parents=True)
            output_dir.mkdir(parents=True)
            source = input_dir / "synthetic_fixture.mkv"
            write_synthetic_av_fixture(source)
            fixture_base = source.stem
            (input_dir / f"{fixture_base}.info.json").write_text(
                json.dumps({"id": fixture_base, "title": "Generated test fixture"}),
                encoding="utf-8",
            )
            (input_dir / f"{fixture_base}.webp").write_bytes(b"synthetic-thumbnail")

            messages = queue.Queue()
            task = EncodeTask(
                channel_id="TEST_CHANNEL",
                video_path=str(source),
                need_video=True,
                need_audio=True,
            )
            success, elapsed = process_one_file(
                task=task,
                progress_q=messages,
                wid=0,
                channel_out=output_dir,
                encoder=FakeNeuralEncoder(),
                enc_cfg=EncoderCfg(resolution=96, fps=24, ffmpeg_prefetch=2),
                audio_enable=True,
                opus_params={
                    "bitrate": "6k",
                    "frame_ms": 60.0,
                    "complexity": 1,
                    "channels": 1,
                    "vbr": "on",
                },
                finalize_mode="atomic",
            )
            self.assertTrue(success)
            self.assertIsNotNone(elapsed)
            records = []
            while not messages.empty():
                records.append(messages.get_nowait())
            self.assertEqual(records[0]["type"], "worker_task_start")
            self.assertTrue(all(record["channel_id"] == "TEST_CHANNEL" for record in records))
            self.assertEqual(records[-1]["type"], "worker_done")

            state_path = write_channel_inventory(
                channel_id="TEST_CHANNEL",
                input_dir=input_dir,
                output_dir=output_dir,
                source_paths=[source],
                source_extensions={".mkv"},
                audio_required=True,
                metadata_required=True,
                encoder_config={"test": True},
            )
            validation = validate_channel(state_path)
            self.assertTrue(validation.eligible, validation.errors)
            self.assertTrue(source.exists())


if __name__ == "__main__":
    unittest.main()
