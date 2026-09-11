import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import torch

from src.training.config import TrainingConfig
from src.training.data import TrainingDataset, frame_count, load_frame, prepare_data, read_manifest
from src.training.video_data import load_video_clip


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class DirectVideoTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        sources = []
        for split in ("train", "validation"):
            path = self.root / f"{split}.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                            "testsrc2=size=96x64:rate=10", "-frames:v", "40",
                            "-c:v", "libx264", "-g", "10", "-bf", "2", str(path)], check=True)
            sources.append(dict(type="videos", path=str(path), split=split))
        self.config = TrainingConfig(task="image", device="cpu",
            data=dict(sources=sources, video_loading="direct", resize_short_edge=48,
                      manifest=str(self.root / "manifest.jsonl"), cache_root=str(self.root / "cache")),
            stages=[dict(name="test", model="image", mode="float", epochs=1, lr=.001,
                         crop_size=[32, 48])], validation=dict(actual_bitstreams=False))

    def test_direct_clips_match_cached_frames_across_keyframes(self):
        prepare_data(self.config, emit=lambda _: None)
        direct = read_manifest(self.config.data.manifest)[0]
        self.assertFalse((self.root / "cache").exists())
        self.assertEqual(frame_count(direct), 40)
        self.assertEqual(direct["frames"], [])
        self.config.data.video_loading = "cache"
        prepare_data(self.config, emit=lambda _: None)
        cached = read_manifest(self.config.data.manifest)[0]
        for start, count in ((0, 1), (8, 7), (33, 7)):
            actual = load_video_clip(direct, start, count)
            expected = torch.stack([load_frame(cached, i) for i in range(start, start + count)])
            self.assertTrue(torch.equal(actual, expected), (start, count))

    def test_deterministic_sampling_validation_and_source_identity(self):
        prepare_data(self.config, emit=lambda _: None)
        records = read_manifest(self.config.data.manifest)
        stage = copy.deepcopy(self.config.stages[0])
        stage.model, stage.sequence_length = "video", 7
        for split in ("train", "validation"):
            dataset = TrainingDataset(records, stage, self.config.data, 42, split)
            self.assertEqual(dataset[0].shape, (7, 3, 32, 48))
            self.assertTrue(torch.equal(dataset[0], dataset[0]))
        source = Path(records[0]["source"])
        with source.open("ab") as f:
            f.write(b"changed")
        with self.assertRaisesRegex(ValueError, "changed since manifest"):
            read_manifest(self.config.data.manifest)
        with self.assertRaisesRegex(ValueError, "changed since manifest"):
            load_video_clip(records[0], 0, 1)

    def test_short_tail_retries_without_padding_and_short_source_fails(self):
        prepare_data(self.config, emit=lambda _: None)
        record = read_manifest(self.config.data.manifest)[0]
        record.update(duration=20, frame_count=200)
        clip = load_video_clip(record, 190, 7)
        self.assertTrue(torch.equal(clip, load_video_clip(record, 0, 7)))
        with self.assertRaisesRegex(ValueError, "cannot supply 65"):
            load_video_clip(record, 0, 65)

    def test_direct_source_leak_is_rejected_even_with_changed_group_id(self):
        prepare_data(self.config, emit=lambda _: None)
        records = read_manifest(self.config.data.manifest)
        records[1]["source"] = records[0]["source"]
        records[1]["source_stat"] = records[0]["source_stat"]
        Path(self.config.data.manifest).write_text("\n".join(map(json.dumps, records)))
        with self.assertRaisesRegex(ValueError, "leaks"):
            read_manifest(self.config.data.manifest)
