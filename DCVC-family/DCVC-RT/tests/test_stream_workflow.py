from __future__ import annotations

import json
import os
import queue
import subprocess
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from src.cli.encode_workflow import EncoderCfg, append_progress_log, write_bytes_final
from src.cli.stream_workflow import (
    RemoteMedia,
    StreamTask,
    build_ytdlp_stream_command,
    build_stream_inputs,
    collect_stream_tasks,
    find_ytdlp_executable,
    list_stream_tasks,
    process_stream_task,
    resolve_remote_media,
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
    def encode_from_ffmpeg_rawpipe(
        self,
        ffmpeg_proc,
        width,
        height,
        out_path,
        log_dir,
        video_id,
        on_progress=None,
        finalize_mode="atomic",
    ):
        raw = ffmpeg_proc.stdout.read()
        frame_size = width * height * 3 // 2
        frames = len(raw) // frame_size
        if frames <= 0 or len(raw) % frame_size:
            raise RuntimeError("fake stream encoder received incomplete YUV420 frames")
        stream = BytesIO()
        write_sps(stream, {
            "sps_id": 0,
            "height": height,
            "width": width,
            "ec_part": 0,
            "use_ada_i": 0,
        })
        for frame_index in range(frames):
            write_ip(stream, frame_index == 0, 0, 14, b"fake-entropy")
        append_progress_log(log_dir, {
            "video_id": video_id,
            "stage": "stats",
            "event": "frames_encoded",
            "frames": frames,
        })
        write_bytes_final(out_path, stream.getbuffer(), mode=finalize_mode)
        if on_progress:
            on_progress(frames, float(frames), 1.0)


class StreamWorkflowTests(unittest.TestCase):
    def test_mixed_channel_ids_and_id_files_preserve_order_and_deduplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_channel = "UC" + "A" * 22
            direct_channel = "UC" + "B" * 22
            second_channel = "UC" + "C" * 22
            first_file = root / f"{first_channel}.txt"
            second_file = root / f"{second_channel}.txt"
            first_file.write_text("videoid0001\nvideoid0002\n\n", encoding="utf-8")
            second_file.write_text("videoid0002\r\nvideoid0003\r\n", encoding="utf-8")

            inputs = build_stream_inputs(
                [],
                [str(first_file), direct_channel, str(second_file)],
                [],
            )
            self.assertIsInstance(inputs[0], StreamTask)
            self.assertEqual(inputs[0].channel_id, first_channel)
            self.assertEqual(
                inputs[2],
                f"https://www.youtube.com/channel/{direct_channel}/videos",
            )

            with patch("src.cli.stream_workflow.list_stream_tasks") as listing:
                listing.return_value = ([
                    StreamTask(
                        "youtube:videoid0002",
                        "videoid0002",
                        "https://www.youtube.com/watch?v=videoid0002",
                        direct_channel,
                        "youtube",
                    ),
                    StreamTask(
                        "youtube:videoid0004",
                        "videoid0004",
                        "https://www.youtube.com/watch?v=videoid0004",
                        direct_channel,
                        "youtube",
                    ),
                ], [])
                tasks, warnings = collect_stream_tasks("/external/yt-dlp", inputs)

            self.assertEqual(warnings, [])
            self.assertEqual(
                [task.video_id for task in tasks],
                ["videoid0001", "videoid0002", "videoid0004", "videoid0003"],
            )

    def test_youtube_id_file_validates_filename_and_line_format(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid_name = root / "my-list.txt"
            invalid_name.write_text("videoid0001\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "filename"):
                build_stream_inputs([], [str(invalid_name)], [])

            with self.assertRaisesRegex(ValueError, "UC plus 22"):
                build_stream_inputs([], ["not-a-channel-id"], [])

            channel_id = "UC" + "A" * 22
            invalid_line = root / f"{channel_id}.txt"
            invalid_line.write_text("https://youtu.be/videoid0001\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 1"):
                build_stream_inputs([], [str(invalid_line)], [])

    def test_collect_stream_tasks_applies_global_max_to_mixed_inputs(self):
        channel_id = "UC" + "A" * 22
        inputs = [
            StreamTask(
                f"youtube:videoid000{index}",
                f"videoid000{index}",
                f"https://www.youtube.com/watch?v=videoid000{index}",
                channel_id,
                "youtube",
            )
            for index in range(1, 4)
        ]
        tasks, warnings = collect_stream_tasks(
            "/external/yt-dlp",
            inputs,
            max_videos=2,
        )
        self.assertEqual(warnings, [])
        self.assertEqual([task.video_id for task in tasks], ["videoid0001", "videoid0002"])

    def test_finds_ytdlp_beside_conda_executable_without_path_install(self):
        with tempfile.TemporaryDirectory() as directory:
            bin_directory = Path(directory) / "bin"
            bin_directory.mkdir()
            conda = bin_directory / "conda"
            ytdlp = bin_directory / "yt-dlp"
            conda.write_text("#!/bin/sh\n", encoding="utf-8")
            ytdlp.write_text("#!/bin/sh\n", encoding="utf-8")
            ytdlp.chmod(0o755)
            with patch.dict(os.environ, {"CONDA_EXE": str(conda)}), \
                    patch("src.cli.stream_workflow.shutil.which", return_value=None):
                self.assertEqual(find_ytdlp_executable(), str(ytdlp.resolve()))

    @patch("src.cli.stream_workflow._run_ytdlp")
    def test_flat_listing_builds_channel_tasks_without_downloading(self, run_ytdlp):
        entry = {
            "id": "abc123",
            "url": "https://www.youtube.com/watch?v=abc123",
            "channel_id": "UC_TEST",
            "extractor_key": "Youtube",
        }
        run_ytdlp.return_value = (0, json.dumps(entry) + "\n", "")
        tasks, warnings = list_stream_tasks(
            "/external/yt-dlp",
            ["https://www.youtube.com/@example/videos"],
        )
        self.assertEqual(warnings, [])
        self.assertEqual(tasks[0].channel_id, "UC_TEST")
        arguments = run_ytdlp.call_args.args[1]
        self.assertIn("--skip-download", arguments)
        self.assertIn("--flat-playlist", arguments)
        self.assertIn("--remote-components", arguments)
        self.assertIn("youtube:player_client=web_safari", arguments)
        self.assertNotIn("-o", arguments)

    @patch("src.cli.stream_workflow._run_ytdlp")
    def test_resolves_separate_video_and_audio_formats_without_download(self, run_ytdlp):
        metadata = {
            "id": "abc123",
            "requested_formats": [
                {
                    "url": "https://media.invalid/video",
                    "vcodec": "avc1",
                    "acodec": "none",
                    "width": 854,
                    "height": 480,
                    "fps": 30,
                    "http_headers": {"User-Agent": "test-video"},
                },
                {
                    "url": "https://media.invalid/audio",
                    "vcodec": "none",
                    "acodec": "opus",
                    "http_headers": {"User-Agent": "test-audio"},
                },
            ],
        }
        run_ytdlp.return_value = (0, json.dumps(metadata), "")
        task = StreamTask(
            "youtube:abc123",
            "abc123",
            "https://www.youtube.com/watch?v=abc123",
            "UC_TEST",
            "youtube",
        )
        media = resolve_remote_media("/external/yt-dlp", task, 480, None)
        self.assertTrue(media.has_audio)
        self.assertEqual((media.width, media.height, media.fps), (854, 480, 30.0))
        arguments = run_ytdlp.call_args.args[1]
        self.assertIn("--skip-download", arguments)
        self.assertIn("youtube:player_client=web_safari", arguments)
        self.assertNotIn("-o", arguments)

        stream_command = build_ytdlp_stream_command(
            "/external/yt-dlp",
            task.webpage_url,
            media.video_format_selector,
            None,
        )
        self.assertIn("--output", stream_command)
        self.assertEqual(stream_command[stream_command.index("--output") + 1], "-")
        self.assertIn("youtube:player_client=web_safari", stream_command)

    def test_synthetic_stream_creates_only_final_archive_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "synthetic-source.mkv"
            output_root = root / "output"
            write_synthetic_av_fixture(source)
            fake_ytdlp = root / "yt-dlp"
            fake_ytdlp.write_text(
                "#!/bin/sh\nexec cat \"$FAKE_STREAM_MEDIA\"\n",
                encoding="utf-8",
            )
            fake_ytdlp.chmod(0o755)
            original_size = source.stat().st_size
            metadata = {
                "id": "stream_fixture",
                "upload_date": "20260902",
                "channel_id": "UC_TEST",
                "title": "Generated stream fixture",
            }
            media = RemoteMedia(
                metadata=metadata,
                width=176,
                height=96,
                fps=24,
                has_audio=True,
                video_format_selector="bestvideo/best",
                audio_format_selector="bestaudio/best",
            )
            task = StreamTask(
                "test:stream_fixture",
                "stream_fixture",
                "https://example.invalid/watch/stream_fixture",
                "UC_TEST",
                "test",
            )
            messages = queue.Queue()
            with patch("src.cli.stream_workflow.resolve_remote_media", return_value=media), \
                    patch.dict(os.environ, {"FAKE_STREAM_MEDIA": str(source)}):
                success, elapsed = process_stream_task(
                    task=task,
                    progress_q=messages,
                    worker_id=0,
                    output_root=output_root,
                    encoder=FakeNeuralEncoder(),
                    config=EncoderCfg(resolution=96, fps=24, ffmpeg_prefetch=2),
                    audio_enabled=True,
                    opus_params={
                        "bitrate": "6k",
                        "frame_ms": 60.0,
                        "complexity": 1,
                        "channels": 1,
                        "vbr": "on",
                    },
                    finalize_mode="atomic",
                    ytdlp_executable=str(fake_ytdlp),
                    cookies_path=None,
                    source_max_height=480,
                    write_thumbnail=False,
                )

            self.assertTrue(success)
            self.assertGreater(elapsed, 0)
            self.assertTrue(source.exists())
            self.assertEqual(source.stat().st_size, original_size)
            channel_output = output_root / "UC_TEST"
            self.assertEqual(len(list(channel_output.glob("*.bin"))), 1)
            self.assertEqual(len(list(channel_output.glob("*.opus"))), 1)
            self.assertEqual(len(list(channel_output.glob("*.info.json"))), 1)
            self.assertEqual(list(channel_output.glob("*.mkv")), [])
            records = []
            while not messages.empty():
                records.append(messages.get_nowait())
            self.assertEqual(records[0]["type"], "worker_task_start")
            worker_start = next(record for record in records if record["type"] == "worker_start")
            self.assertEqual(worker_start["resolution"], "176, 96 -> 176, 96")
            self.assertEqual(records[-1]["type"], "worker_done")

    def test_single_video_rescue_reuses_existing_archived_basename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "synthetic-source.mkv"
            output_root = root / "output"
            channel_output = output_root / "UC_TEST"
            channel_output.mkdir(parents=True)
            write_synthetic_av_fixture(source)
            fake_ytdlp = root / "yt-dlp"
            fake_ytdlp.write_text(
                "#!/bin/sh\nexec cat \"$FAKE_STREAM_MEDIA\"\n",
                encoding="utf-8",
            )
            fake_ytdlp.chmod(0o755)
            archived_base = "stream_fixture_1615820413"
            archived_metadata = {
                "id": "stream_fixture",
                "title": "Archived metadata must be preserved",
            }
            info_path = channel_output / f"{archived_base}.info.json"
            info_path.write_text(json.dumps(archived_metadata), encoding="utf-8")
            media = RemoteMedia(
                metadata={
                    "id": "stream_fixture",
                    "upload_date": "20210315",
                    "channel_id": "UC_TEST",
                    "title": "Fresh remote metadata",
                },
                width=176,
                height=96,
                fps=24,
                has_audio=True,
                video_format_selector="bestvideo/best",
                audio_format_selector="bestaudio/best",
            )
            task = StreamTask(
                "test:stream_fixture",
                "stream_fixture",
                "https://example.invalid/watch/stream_fixture",
                "UC_TEST",
                "test",
            )
            messages = queue.Queue()

            with patch("src.cli.stream_workflow.resolve_remote_media", return_value=media), \
                    patch.dict(os.environ, {"FAKE_STREAM_MEDIA": str(source)}):
                success, _ = process_stream_task(
                    task=task,
                    progress_q=messages,
                    worker_id=0,
                    output_root=output_root,
                    encoder=FakeNeuralEncoder(),
                    config=EncoderCfg(resolution=96, fps=24, ffmpeg_prefetch=2),
                    audio_enabled=False,
                    opus_params={},
                    finalize_mode="atomic",
                    ytdlp_executable=str(fake_ytdlp),
                    cookies_path=None,
                    source_max_height=480,
                    write_thumbnail=False,
                )

            self.assertTrue(success)
            self.assertEqual(json.loads(info_path.read_text(encoding="utf-8")), archived_metadata)
            self.assertEqual(len(list(channel_output.glob(f"{archived_base}_*.bin"))), 1)
            self.assertFalse((channel_output / "stream_fixture_20210315.info.json").exists())

    def test_failed_downloader_does_not_publish_partial_bitstream(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "synthetic-source.mkv"
            output_root = root / "output"
            write_synthetic_av_fixture(source)
            fake_ytdlp = root / "yt-dlp"
            fake_ytdlp.write_text(
                "#!/bin/sh\ncat \"$FAKE_STREAM_MEDIA\"\nexit 17\n",
                encoding="utf-8",
            )
            fake_ytdlp.chmod(0o755)
            media = RemoteMedia(
                metadata={
                    "id": "failed_fixture",
                    "upload_date": "20260902",
                    "channel_id": "UC_TEST",
                },
                width=176,
                height=96,
                fps=24,
                has_audio=True,
                video_format_selector="bestvideo/best",
                audio_format_selector="bestaudio/best",
            )
            task = StreamTask(
                "test:failed_fixture",
                "failed_fixture",
                "https://example.invalid/watch/failed_fixture",
                "UC_TEST",
                "test",
            )
            messages = queue.Queue()
            with patch("src.cli.stream_workflow.resolve_remote_media", return_value=media), \
                    patch.dict(os.environ, {"FAKE_STREAM_MEDIA": str(source)}):
                success, elapsed = process_stream_task(
                    task=task,
                    progress_q=messages,
                    worker_id=0,
                    output_root=output_root,
                    encoder=FakeNeuralEncoder(),
                    config=EncoderCfg(resolution=96, fps=24, ffmpeg_prefetch=2),
                    audio_enabled=False,
                    opus_params={},
                    finalize_mode="atomic",
                    ytdlp_executable=str(fake_ytdlp),
                    cookies_path=None,
                    source_max_height=480,
                    write_thumbnail=False,
                )

            self.assertFalse(success)
            self.assertIsNone(elapsed)
            channel_output = output_root / "UC_TEST"
            self.assertEqual(list(channel_output.glob("*.bin")), [])
            self.assertEqual(list(channel_output.glob(".*.stream.*")), [])
            records = []
            while not messages.empty():
                records.append(messages.get_nowait())
            self.assertEqual(records[-1]["type"], "worker_fail")
            self.assertIn("rc=17", records[-1]["error"])
            progress_records = [
                json.loads(line)
                for line in (channel_output / ".progress.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(progress_records[-1]["status"], "failed")
            self.assertEqual(progress_records[-1]["stage"], "stream")
            self.assertIn("rc=17", progress_records[-1]["error"])


if __name__ == "__main__":
    unittest.main()
