import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from src.models.architecture import ImageArchitecture, VideoArchitecture
from src.training.config import TrainingConfig
from src.training.data import TrainingDataset, load_frame, prepare_data, read_manifest
from src.training.trainer import Trainer, train


def fixture_config(root, *, mode="float", actual=False):
    sources = []
    for index, split in enumerate(("train", "train", "validation")):
        sequence = root / f"sequence-{index}"
        sequence.mkdir()
        for frame in range(6):
            y, x = np.mgrid[:40, :56]
            pixels = np.stack(((x * 3 + frame * 7) % 256, (y * 4 + index * 15) % 256,
                               (x + y + frame * 5) % 256), axis=-1).astype(np.uint8)
            Image.fromarray(pixels).save(sequence / f"im{frame + 1}.png")
        sources.append({"type": "frame_sequences", "path": str(sequence), "split": split})
    image = {key: 8 if "channels" in key else 1 for key in ImageArchitecture().to_dict()}
    video = {key: 8 if "channels" in key else 1 for key in VideoArchitecture().to_dict()}
    stages = [{"name": "image", "model": "image", "mode": mode, "epochs": 1, "lr": 1e-3,
               "crop_size": [32, 48], "samples_per_epoch": 4},
              {"name": "video", "model": "video", "mode": mode, "epochs": 1, "lr": 1e-3,
               "crop_size": [32, 48], "samples_per_epoch": 4, "sequence_length": 4,
               "temporal_gradient_length": 2}]
    return TrainingConfig(task="pair", device="cuda" if mode == "int16" else "cpu", seed=13,
                          model={"image": image, "video": video}, stages=stages,
                          data={"manifest": str(root / "manifest.jsonl"), "cache_root": str(root / "cache"),
                                "sources": sources, "num_workers": 0, "batch_per_gpu": 1},
                          optimizer={"accumulation_steps": 2},
                          validation={"actual_bitstreams": actual, "qps": [0, 63], "max_samples": 1,
                                      "max_frames": 6, "every_steps": 2, "reset_interval": 3},
                          output={"root": str(root / "run"), "checkpoint_every_steps": 1, "log_every_steps": 1})


class DataTests(unittest.TestCase):
    def test_manifest_relative_paths_and_changed_source_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = fixture_config(root)
            prepare_data(config, emit=lambda text: None)
            records = read_manifest(config.data.manifest)
            for record in records:
                record["frames"] = [str(Path(frame).relative_to(root)) for frame in record["frames"]]
            Path(config.data.manifest).write_text("\n".join(json.dumps(row) for row in records))
            loaded = read_manifest(config.data.manifest)
            frame = Path(loaded[0]["frames"][0])
            self.assertTrue(frame.is_absolute())
            with frame.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "changed since manifest creation"):
                read_manifest(config.data.manifest)

    def test_preparation_and_sampling_are_source_disjoint_and_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture_config(Path(directory))
            prepare_data(config, emit=lambda text: None)
            records = read_manifest(config.data.manifest)
            train_groups = {r["source_group"] for r in records if r["split"] == "train"}
            validation_groups = {r["source_group"] for r in records if r["split"] == "validation"}
            self.assertFalse(train_groups & validation_groups)
            dataset = TrainingDataset(records, config.stages[1], config.data, config.seed)
            self.assertTrue(torch.equal(dataset[0], dataset[0]))
            self.assertEqual(dataset[0].shape, (4, 3, 32, 48))
            changed = copy.deepcopy(records)
            changed[-1]["frames"][0] = changed[0]["frames"][0]
            Path(config.data.manifest).write_text("\n".join(json.dumps(row) for row in changed))
            with self.assertRaisesRegex(ValueError, "leaks"):
                read_manifest(config.data.manifest)

    def test_original_video_preparation_and_cache_reuse(self):
        import subprocess
        if not shutil.which("ffmpeg"):
            self.skipTest("FFmpeg is required")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = fixture_config(root)
            video = root / "source.mkv"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=56x40:rate=4",
                            "-frames:v", "6", "-c:v", "ffv1", str(video)], check=True)
            config.data.sources[0].type = "videos"
            config.data.sources[0].path = str(video)
            prepare_data(config, emit=lambda text: None)
            record = next(r for r in read_manifest(config.data.manifest) if r["format"] == "yuv420_npz")
            timestamps = [Path(path).stat().st_mtime_ns for path in record["frames"]]
            prepare_data(config, emit=lambda text: None)
            self.assertEqual(timestamps, [Path(path).stat().st_mtime_ns for path in record["frames"]])
            self.assertEqual(load_frame(record, 0).shape, (3, 40, 56))
            self.assertTrue(video.exists())


class TrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_pair_training_and_exact_optimizer_boundary_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = fixture_config(root)
            prepare_data(config, emit=lambda text: None)
            saved = root / "intermediate.pt"
            original_save = Trainer.save
            def capture(trainer, stage, filename="last.pt"):
                original_save(trainer, stage, filename)
                if trainer.global_step == 1 and filename == "last.pt" and not saved.exists():
                    shutil.copyfile(trainer.root / filename, saved)
            with contextlib.redirect_stdout(io.StringIO()), patch.object(Trainer, "save", capture):
                self.assertEqual(train(config), 0)
            uninterrupted = torch.load(root / "run/last.pt", map_location="cpu", weights_only=True)
            resumed_config = copy.deepcopy(config)
            resumed_config.output.root = str(root / "resumed")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(train(resumed_config, str(saved)), 0)
            resumed = torch.load(root / "resumed/last.pt", map_location="cpu", weights_only=True)
            self.assertEqual(uninterrupted["global_step"], 4)
            self.assertEqual(uninterrupted["global_step"], resumed["global_step"])
            for kind in ("image", "video"):
                for key, tensor in uninterrupted["models"][kind].items():
                    self.assertTrue(torch.equal(tensor, resumed["models"][kind][key]), (kind, key))
            self.assertTrue((root / "run/best-image.pt").is_file())
            self.assertTrue((root / "run/best-video.pt").is_file())

    def test_two_rank_ddp_accumulation_and_nonfinite_skip_keep_models_synchronized(self):
        python = os.environ.get("DCVC_TEST_DDP_PYTHON", sys.executable)
        if not torch.distributed.is_available() and "DCVC_TEST_DDP_PYTHON" not in os.environ:
            self.skipTest("this PyTorch build lacks distributed support; set DCVC_TEST_DDP_PYTHON to a DDP-enabled interpreter")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = fixture_config(root)
            for stage in config.stages:
                stage.samples_per_epoch = 8
            prepare_data(config, emit=lambda text: None)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config.to_dict()))
            script = root / "worker.py"
            script.write_text('''import hashlib, sys
from src.training.config import load_config
from src.training.trainer import Trainer
from src.training.data import atomic_json
def model_hashes(models):
    hashes = {}
    for kind, model in models.items():
        h = hashlib.sha256()
        for key, tensor in model.state_dict().items():
            h.update(key.encode())
            h.update(tensor.detach().cpu().numpy().tobytes())
        hashes[kind] = h.hexdigest()
    return hashes
class CheckedTrainer(Trainer):
    def enter_stage(self, stage):
        super().enter_stage(stage)
        if self.global_step == 0:
            self.initial_hashes = model_hashes(self.models)
            if self.rank == 1:
                parameter = next(self.models[stage.model].parameters())
                parameter.register_hook(lambda grad: grad * float("nan") if self.global_step == 0 else grad)
    def log(self, event, **values):
        super().log(event, **values)
        if event == "train_step" and self.global_step == 1:
            assert values["skipped"], "one rank's NaN must skip all ranks"
            assert model_hashes(self.models) == self.initial_hashes, "skipped update changed weights"
        if event == "training_completed":
            atomic_json(self.root / ("rank-%d.json" % self.rank), model_hashes(self.models))
raise SystemExit(CheckedTrainer(load_config(sys.argv[1])).run())
''')
            project = str(Path(__file__).resolve().parents[1])
            env = dict(os.environ, PYTHONPATH=project, OMP_NUM_THREADS="1")
            result = subprocess.run([python, "-m", "torch.distributed.run", "--standalone",
                                     "--nproc_per_node=2", str(script), str(config_path)],
                                    cwd=project, env=env, capture_output=True, text=True, timeout=110)
            self.assertEqual(result.returncode, 0, result.stdout[-4000:] + result.stderr[-6000:])
            first = json.loads((root / "run/rank-0.json").read_text())
            second = json.loads((root / "run/rank-1.json").read_text())
            self.assertEqual(first, second)
            state = torch.load(root / "run/last.pt", map_location="cpu", weights_only=True)
            self.assertEqual(state["world_size"], 2)
            self.assertEqual(len(state["rng_states"]), 2)
            rows = [json.loads(line) for line in (root / "run/metrics.jsonl").read_text().splitlines()]
            skipped = [row for row in rows if row.get("skipped")]
            self.assertEqual(len(skipped), 1)
            self.assertIsNone(skipped[0]["gradient_norm"])


if __name__ == "__main__":
    unittest.main()
