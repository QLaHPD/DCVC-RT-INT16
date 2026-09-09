from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from src.cli.encode_workflow import _publish_shared_result, shared_stage_path
from src.cli.progress import build_encode_output_index
from src.cli.shared_work import SharedWorkError, SharedWorkPool


class SharedWorkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.channel = Path(self.temporary.name) / "CHANNEL"
        self.channel.mkdir()
        self.config = {"resolution": 96, "qp_i": 35, "qp_p": 14}
        self.first = SharedWorkPool(lease_seconds=30, instance_name="first")
        self.second = SharedWorkPool(lease_seconds=30, instance_name="second")
        self.first.ensure_pipeline(self.channel, self.config)
        self.second.ensure_pipeline(self.channel, self.config)

    def tearDown(self):
        self.temporary.cleanup()

    def test_only_one_instance_can_claim_a_video(self):
        lease, info = self.first.try_acquire(self.channel, "video")
        self.assertIsNotNone(lease)
        self.assertIsNone(info)
        other, owner = self.second.try_acquire(self.channel, "video")
        self.assertIsNone(other)
        self.assertIn("first", owner.owner)
        lease.release()
        replacement, _ = self.second.try_acquire(self.channel, "video")
        self.assertIsNotNone(replacement)
        replacement.release()

    def test_heartbeat_progress_is_visible_to_another_instance(self):
        lease, _ = self.first.try_acquire(self.channel, "video")
        lease.heartbeat({"status": "encoding", "frames": 123, "fps": 8.5})
        info = self.second.read_claim(self.channel, "video")
        self.assertEqual(info.progress["frames"], 123)
        self.assertEqual(info.progress["fps"], 8.5)
        lease.release()

    def test_stale_claim_is_recovered_and_old_owner_is_fenced(self):
        lease, _ = self.first.try_acquire(self.channel, "video")
        self.assertFalse(lease.recovered_stale)
        state_path = lease.claim_path / "state.json"
        old = time.time() - 60
        os.utime(state_path, (old, old))
        replacement, _ = self.second.try_acquire(self.channel, "video")
        self.assertIsNotNone(replacement)
        self.assertTrue(replacement.recovered_stale)
        with self.assertRaisesRegex(SharedWorkError, "was lost"):
            lease.assert_owned()
        lease.release()
        replacement.release()

    def test_done_marker_requires_and_tracks_real_artifacts(self):
        lease, _ = self.first.try_acquire(self.channel, "video")
        artifact = self.channel / "video_176x96_qI35_qP14.bin"
        artifact.write_bytes(b"bitstream")
        lease.mark_done([artifact.name])
        lease.release()
        self.assertEqual(
            self.second.completed_artifacts(self.channel, "video"),
            (artifact.name,),
        )
        artifact.unlink()
        self.assertIsNone(self.second.completed_artifacts(self.channel, "video"))

    def test_different_pipeline_settings_are_rejected(self):
        incompatible = SharedWorkPool(lease_seconds=30)
        with self.assertRaisesRegex(SharedWorkError, "settings differ"):
            incompatible.ensure_pipeline(self.channel, {**self.config, "qp_p": 20})

    def test_only_current_lease_can_publish_staged_output(self):
        lease, _ = self.first.try_acquire(self.channel, "video")
        output_index = build_encode_output_index(self.channel, parse_progress=False)
        staged = shared_stage_path(self.channel, "video", lease.token, ".bin")
        staged.write_bytes(b"complete-bitstream")
        final = self.channel / "video_176x96_qI35_qP14.bin"
        artifacts = _publish_shared_result(
            self.channel,
            "video",
            [{"temporary": str(staged), "final": str(final)}],
            lease,
            audio_enabled=False,
            output_index=output_index,
        )
        self.assertEqual(artifacts, [final.name])
        self.assertEqual(final.read_bytes(), b"complete-bitstream")
        self.assertEqual(self.first.completed_artifacts(self.channel, "video"), (final.name,))
        self.assertEqual(output_index.artifact_names("video", False), [final.name])
        lease.release()

    def test_expired_owner_cannot_publish_after_takeover(self):
        lease, _ = self.first.try_acquire(self.channel, "video")
        staged = shared_stage_path(self.channel, "video", lease.token, ".bin")
        staged.write_bytes(b"old-worker-output")
        state_path = lease.claim_path / "state.json"
        old = time.time() - 60
        os.utime(state_path, (old, old))
        replacement, _ = self.second.try_acquire(self.channel, "video")
        final = self.channel / "video_176x96_qI35_qP14.bin"
        with self.assertRaisesRegex(SharedWorkError, "was lost"):
            _publish_shared_result(
                self.channel,
                "video",
                [{"temporary": str(staged), "final": str(final)}],
                lease,
                audio_enabled=False,
            )
        self.assertFalse(final.exists())
        replacement.release()
        lease.release()


if __name__ == "__main__":
    unittest.main()
