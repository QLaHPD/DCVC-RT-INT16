from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from src.cli.channel_lifecycle import (
    ARCHIVE_MANIFEST_FILENAME,
    AUDIT_FILENAME,
    LifecycleError,
    cleanup_channel,
    validate_channel,
    write_channel_inventory,
)
from src.cli.lifecycle_runtime import ChannelLifecycleController, ChannelRuntimeState
from src.utils.stream_helper import write_ip, write_sps


SOURCE_EXTENSIONS = {".mkv", ".webm", ".mp4"}


def write_valid_bitstream(path: Path, frames: int = 1):
    with path.open("wb") as f:
        write_sps(f, {
            "sps_id": 0,
            "height": 96,
            "width": 176,
            "ec_part": 0,
            "use_ada_i": 0,
        })
        for index in range(frames):
            write_ip(f, index == 0, 0, 14, b"test-entropy-payload")


class LifecycleFixture:
    def __init__(self, root: Path, count: int = 3, metadata: bool = True):
        self.input_dir = root / "incoming" / "TEST_CHANNEL"
        self.output_dir = root / "encoded" / "TEST_CHANNEL"
        self.input_dir.mkdir(parents=True)
        self.output_dir.mkdir(parents=True)
        self.sources = []
        self.bases = []
        extensions = [".mkv", ".webm", ".mp4"]
        progress = []
        for index in range(count):
            base = f"video_{index:02d}"
            source = self.input_dir / f"{base}{extensions[index % len(extensions)]}"
            source.write_bytes((f"original-{index}-" * 20).encode())
            if metadata:
                (self.input_dir / f"{base}.info.json").write_text(
                    json.dumps({"id": base, "title": f"Fixture {index}"}), encoding="utf-8"
                )
            (self.input_dir / f"{base}.webp").write_bytes(b"thumbnail")
            bin_path = self.output_dir / f"{base}_176x96_qI35_qP14.bin"
            write_valid_bitstream(bin_path, frames=2)
            (self.output_dir / f"{base}.opus").write_bytes(b"synthetic-opus-for-mocked-probe")
            progress.extend([
                {"video_id": base, "stage": "stats", "event": "frames_encoded", "frames": 2},
                {"video_id": base, "status": "video-done", "out_bin": str(bin_path)},
            ])
            self.sources.append(source)
            self.bases.append(base)
        (self.input_dir / "unrelated.keep").write_text("must survive", encoding="utf-8")
        with (self.output_dir / ".progress.jsonl").open("w", encoding="utf-8") as f:
            for record in progress:
                f.write(json.dumps(record) + "\n")
        self.state_path = write_channel_inventory(
            channel_id="TEST_CHANNEL",
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            source_paths=self.sources,
            source_extensions=SOURCE_EXTENSIONS,
            audio_required=True,
            metadata_required=True,
            encoder_config={"qp_i": 35, "qp_p": 14},
        )


class ChannelLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    @mock.patch("src.cli.channel_lifecycle._probe_opus", return_value=1.0)
    def test_exact_cleanup_deletes_only_three_original_videos(self, _probe):
        fixture = LifecycleFixture(self.root)
        validation = validate_channel(fixture.state_path)
        self.assertTrue(validation.eligible, validation.errors)
        self.assertEqual(len(validation.items), 3)

        result = cleanup_channel(fixture.state_path, actor="unit-test")
        self.assertTrue(result.success, result.errors)
        self.assertEqual(result.deleted_files, 3)
        for source in fixture.sources:
            self.assertFalse(source.exists())
        self.assertTrue((fixture.input_dir / "unrelated.keep").is_file())
        for base in fixture.bases:
            self.assertTrue((fixture.input_dir / f"{base}.info.json").is_file())
            self.assertTrue((fixture.input_dir / f"{base}.webp").is_file())
            self.assertTrue((fixture.output_dir / f"{base}.info.json").is_file())
            self.assertTrue((fixture.output_dir / f"{base}.webp").is_file())
            self.assertTrue((fixture.output_dir / f"{base}.opus").is_file())
            self.assertEqual(len(list(fixture.output_dir.glob(f"{base}_*.bin"))), 1)
        self.assertTrue((fixture.output_dir / ARCHIVE_MANIFEST_FILENAME).is_file())
        self.assertTrue((fixture.output_dir / AUDIT_FILENAME).is_file())

    @mock.patch("src.cli.channel_lifecycle._probe_opus", return_value=1.0)
    def test_changed_source_blocks_cleanup(self, _probe):
        fixture = LifecycleFixture(self.root, count=1)
        fixture.sources[0].write_bytes(b"replacement contents")
        result = cleanup_channel(fixture.state_path, actor="unit-test")
        self.assertFalse(result.success)
        self.assertTrue(fixture.sources[0].exists())
        self.assertTrue(any("source changed" in error for error in result.errors))

    @mock.patch("src.cli.channel_lifecycle._probe_opus", return_value=1.0)
    def test_missing_metadata_blocks_cleanup(self, _probe):
        fixture = LifecycleFixture(self.root, count=1, metadata=False)
        result = cleanup_channel(fixture.state_path, actor="unit-test")
        self.assertFalse(result.success)
        self.assertTrue(fixture.sources[0].exists())
        self.assertTrue(any("missing required metadata" in error for error in result.errors))

    @mock.patch("src.cli.channel_lifecycle._probe_opus", return_value=1.0)
    def test_dry_run_writes_manifest_without_deleting(self, _probe):
        fixture = LifecycleFixture(self.root)
        result = cleanup_channel(fixture.state_path, dry_run=True, actor="unit-test")
        self.assertTrue(result.success, result.errors)
        self.assertTrue(result.dry_run)
        self.assertEqual(result.deleted_files, 0)
        self.assertTrue(all(source.exists() for source in fixture.sources))
        self.assertTrue((fixture.output_dir / ARCHIVE_MANIFEST_FILENAME).is_file())

    @mock.patch("src.cli.channel_lifecycle._probe_opus", return_value=1.0)
    def test_prevalidated_cleanup_uses_fast_identity_check(self, _probe):
        fixture = LifecycleFixture(self.root, count=1)
        validation = validate_channel(fixture.state_path)
        self.assertTrue(validation.eligible, validation.errors)

        with mock.patch(
            "src.cli.channel_lifecycle.validate_channel",
            side_effect=AssertionError("full validation must not run again"),
        ):
            result = cleanup_channel(
                fixture.state_path,
                dry_run=True,
                actor="unit-test",
                prevalidated=validation,
            )

        self.assertTrue(result.success, result.errors)
        self.assertTrue(fixture.sources[0].exists())

    @mock.patch("src.cli.channel_lifecycle._probe_opus", return_value=1.0)
    def test_prevalidated_cleanup_stops_if_retained_artifact_changed(self, _probe):
        fixture = LifecycleFixture(self.root, count=1)
        validation = validate_channel(fixture.state_path)
        bitstream = next(fixture.output_dir.glob("*.bin"))
        metadata = bitstream.stat()
        os.utime(
            bitstream,
            ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1_000_000_000),
        )

        result = cleanup_channel(
            fixture.state_path,
            actor="unit-test",
            prevalidated=validation,
        )

        self.assertFalse(result.success)
        self.assertTrue(fixture.sources[0].exists())
        self.assertTrue(any("retained artifact changed" in error for error in result.errors))

    @mock.patch("src.cli.channel_lifecycle._probe_opus", return_value=1.0)
    def test_cleanup_stops_while_shared_work_claim_is_active(self, _probe):
        fixture = LifecycleFixture(self.root, count=1)
        validation = validate_channel(fixture.state_path)
        claim = fixture.output_dir / ".dcvc-shared-work" / "claims" / "active-job"
        claim.mkdir(parents=True)
        (claim / "state.json").write_text("{}\n", encoding="utf-8")

        result = cleanup_channel(
            fixture.state_path,
            actor="unit-test",
            prevalidated=validation,
        )

        self.assertFalse(result.success)
        self.assertTrue(fixture.sources[0].exists())
        self.assertTrue(any("shared-work claim" in error for error in result.errors))

    @mock.patch("src.cli.channel_lifecycle._probe_opus", return_value=1.0)
    def test_partial_failure_is_restart_safe(self, _probe):
        fixture = LifecycleFixture(self.root)
        original_unlink = __import__("os").unlink
        failed_once = {"value": False}

        def fail_second_source(path, *args, **kwargs):
            if path == fixture.sources[1].name and not failed_once["value"]:
                failed_once["value"] = True
                raise OSError("injected unlink failure")
            return original_unlink(path, *args, **kwargs)

        with mock.patch("src.cli.channel_lifecycle.os.unlink", fail_second_source):
            first = cleanup_channel(fixture.state_path, actor="unit-test")
        self.assertFalse(first.success)
        self.assertEqual(first.deleted_files, 1)
        self.assertFalse(fixture.sources[0].exists())
        self.assertTrue(fixture.sources[1].exists())

        resumed = cleanup_channel(fixture.state_path, actor="unit-test-resume")
        self.assertTrue(resumed.success, resumed.errors)
        self.assertFalse(any(source.exists() for source in fixture.sources))

    @mock.patch("src.cli.channel_lifecycle._probe_opus", return_value=1.0)
    def test_crash_after_unlink_before_completion_audit_is_recoverable(self, _probe):
        fixture = LifecycleFixture(self.root, count=1)
        from src.cli import channel_lifecycle
        original_append = channel_lifecycle._append_audit
        failed_once = {"value": False}

        def fail_post_unlink(output_dir, event, **fields):
            if event == "source-deleted" and not failed_once["value"]:
                failed_once["value"] = True
                raise OSError("injected audit failure after unlink")
            return original_append(output_dir, event, **fields)

        with mock.patch("src.cli.channel_lifecycle._append_audit", fail_post_unlink):
            first = cleanup_channel(fixture.state_path, actor="unit-test")
        self.assertFalse(first.success)
        self.assertFalse(fixture.sources[0].exists())

        resumed = cleanup_channel(fixture.state_path, actor="unit-test-resume")
        self.assertTrue(resumed.success, resumed.errors)
        self.assertEqual(resumed.deleted_files, 0)

    @mock.patch("src.cli.channel_lifecycle._probe_opus", return_value=1.0)
    def test_truncated_bitstream_blocks_cleanup(self, _probe):
        fixture = LifecycleFixture(self.root, count=1)
        bitstream = next(fixture.output_dir.glob("*.bin"))
        bitstream.write_bytes(bitstream.read_bytes()[:-2])
        result = cleanup_channel(fixture.state_path, actor="unit-test")
        self.assertFalse(result.success)
        self.assertTrue(fixture.sources[0].exists())
        self.assertTrue(any("bitstream" in error or "truncated" in error for error in result.errors))

    @mock.patch("src.cli.channel_lifecycle._probe_opus", return_value=1.0)
    def test_shared_progress_path_is_rebased_to_current_output_directory(self, _probe):
        fixture = LifecycleFixture(self.root, count=1)
        progress_path = fixture.output_dir / ".progress.jsonl"
        records = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines()]
        for record in records:
            if record.get("status") == "video-done":
                record["out_bin"] = str(
                    Path("/remote/machine/mount/YOUTUBE/TEST_CHANNEL") / Path(record["out_bin"]).name
                )
        progress_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )

        validation = validate_channel(fixture.state_path)

        self.assertTrue(validation.eligible, validation.errors)
        bitstreams = [
            artifact.path
            for item in validation.items
            for artifact in item.artifacts
            if artifact.kind == "bitstream"
        ]
        self.assertEqual(bitstreams, [str(next(fixture.output_dir.glob("*.bin")).absolute())])

    @mock.patch("src.cli.channel_lifecycle._probe_opus", return_value=1.0)
    def test_later_failed_progress_record_invalidates_old_success(self, _probe):
        fixture = LifecycleFixture(self.root, count=1)
        with (fixture.output_dir / ".progress.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "video_id": fixture.bases[0], "status": "failed", "stage": "video"
            }) + "\n")
        result = cleanup_channel(fixture.state_path, actor="unit-test")
        self.assertFalse(result.success)
        self.assertTrue(fixture.sources[0].exists())
        self.assertTrue(any("no completed bitstream record" in error for error in result.errors))

    def test_channel_traversal_is_rejected(self):
        with self.assertRaises(LifecycleError):
            write_channel_inventory(
                channel_id="../escape",
                input_dir=self.root,
                output_dir=self.root / "out",
                source_paths=[],
                source_extensions=SOURCE_EXTENSIONS,
                audio_required=True,
                metadata_required=True,
                encoder_config={},
            )

    @mock.patch("src.cli.channel_lifecycle._probe_opus", return_value=1.0)
    def test_controller_waits_for_approval_without_blocking_other_channel(self, _probe):
        first = LifecycleFixture(self.root / "first", count=1)
        second = LifecycleFixture(self.root / "second", count=1)
        channels = [
            ChannelRuntimeState("FIRST", first.state_path, total=1, already_done=1, pending=0),
            ChannelRuntimeState("SECOND", second.state_path, total=1, already_done=0, pending=1),
        ]
        controller = ChannelLifecycleController(channels, policy="prompt")
        controller.mark_started("SECOND", second.bases[0])

        deadline = time.time() + 5
        while time.time() < deadline and controller.channels["FIRST"].status == "validating":
            controller.poll()
            time.sleep(0.01)
        self.assertEqual(controller.channels["FIRST"].status, "awaiting-approval")
        self.assertEqual(controller.channels["SECOND"].status, "encoding")
        controller.mark_terminal("SECOND", second.bases[0], success=True)

        deadline = time.time() + 5
        while time.time() < deadline and controller.channels["SECOND"].status == "validating":
            controller.poll()
            time.sleep(0.01)
        self.assertEqual(controller.channels["SECOND"].status, "awaiting-approval")
        self.assertTrue(all(source.exists() for source in first.sources + second.sources))
        controller.keep("FIRST")
        controller.keep("SECOND")
        controller.close()

    def test_worker_failure_never_schedules_cleanup(self):
        fixture = LifecycleFixture(self.root, count=1)
        controller = ChannelLifecycleController([
            ChannelRuntimeState("TEST_CHANNEL", fixture.state_path, total=1, already_done=0, pending=1)
        ], policy="auto")
        controller.mark_started("TEST_CHANNEL", fixture.bases[0])
        controller.mark_terminal("TEST_CHANNEL", fixture.bases[0], success=False, error="injected failure")
        controller.poll()
        self.assertEqual(controller.channels["TEST_CHANNEL"].status, "encode-failed")
        self.assertFalse(controller.has_background_work())
        self.assertTrue(fixture.sources[0].exists())
        controller.close()

    def test_validation_failure_remains_actionable_in_prompt_mode(self):
        fixture = LifecycleFixture(self.root, count=1)
        next(fixture.output_dir.glob("*.bin")).unlink()
        controller = ChannelLifecycleController([
            ChannelRuntimeState("TEST_CHANNEL", fixture.state_path, total=1, already_done=1, pending=0)
        ], policy="prompt")

        deadline = time.time() + 5
        while time.time() < deadline and controller.channels["TEST_CHANNEL"].status == "validating":
            controller.poll()
            time.sleep(0.01)

        self.assertEqual(controller.channels["TEST_CHANNEL"].status, "validation-failed")
        self.assertEqual(controller.awaiting_actions(), ["TEST_CHANNEL"])
        self.assertEqual(controller.awaiting_approvals(), [])
        controller.close()

    @mock.patch("src.cli.lifecycle_runtime.datetime")
    def test_lifecycle_events_include_local_start_time(self, datetime_mock):
        datetime_mock.now.return_value.astimezone.return_value.isoformat.return_value = (
            "2026-09-09T09:10:03-03:00"
        )
        fixture = LifecycleFixture(self.root, count=1)

        controller = ChannelLifecycleController([
            ChannelRuntimeState("TEST_CHANNEL", fixture.state_path, total=1, already_done=1, pending=0)
        ], policy="prompt")

        self.assertEqual(
            controller.events()[0],
            "[2026-09-09T09:10:03-03:00] TEST_CHANNEL: validating retained artifacts",
        )
        controller.close()


if __name__ == "__main__":
    unittest.main()
