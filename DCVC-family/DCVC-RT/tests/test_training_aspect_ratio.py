import contextlib
import copy
import io
import json
import math
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
from torch.utils.data import DataLoader

from src.models.architecture import ImageArchitecture, create_model
from src.training.aspect_ratio import AspectRatioBatchSampler, nearest_bucket
from src.training.checkpoint import atomic_save, model_checkpoint
from src.training.config import StageConfig, TrainingConfig, ValidationConfig, load_config
from src.training.data import TrainingDataset, load_frame, prepare_data, read_manifest
from src.training.trainer import Trainer, RemainingBatches, _comparable_config, train
from src.training.validation import validation_frames, pad_for_codec, summarize_qps, run_isolated


def fixture(root):
    sources = []
    shapes = [(43, 43), (35, 71), (47, 63), (71, 39)]
    for split in ("train", "validation"):
        directory = root / split / "nested"
        directory.mkdir(parents=True)
        for index, ((h, w), ext) in enumerate(zip(shapes, ("PNG", "jpg", "jpeg", "webp"))):
            y, x = np.mgrid[:h, :w]
            data = np.stack(((x * 3) % 256, (y * 5) % 256, (x + y) % 256), axis=-1).astype(np.uint8)
            Image.fromarray(data).save(directory / f"image{index}.{ext}")
        sources.append(dict(type="images", path=str(root / split), split=split))
    model = {k: 8 if "channels" in k else 1 for k in ImageArchitecture().to_dict()}
    return TrainingConfig(task="image", device="cpu", model=dict(image=model), seed=42,
        data=dict(sources=sources, manifest=str(root / "manifest.jsonl"), batch_per_gpu=2, num_workers=0),
        stages=[dict(name="image", model="image", mode="float", epochs=2, lr=.001,
                     crop_buckets=[[32, 32], [32, 64], [32, 48], [48, 32]], samples_per_epoch=16)],
        optimizer=dict(accumulation_steps=2),
        validation=dict(actual_bitstreams=False, image_mode="full", qps=[0], max_samples=2, every_steps=4),
        output=dict(root=str(root / "run"), checkpoint_every_steps=1, log_every_steps=1))


class AspectRatioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_invalid_shapes_and_video_buckets_are_rejected(self):
        for buckets in ([], [[31, 32]], [[32]], [[32, True]], [[32, 32], [32, 32]]):
            with self.assertRaises(ValueError):
                StageConfig(name="i", model="image", mode="float", epochs=1, lr=.001, crop_buckets=buckets)
        with self.assertRaises(ValueError):
            StageConfig(name="p", model="video", mode="float", epochs=1, lr=.001,
                        sequence_length=2, crop_buckets=[[32, 32]])
        with self.assertRaises(ValueError):
            ValidationConfig(image_mode="crop", image_max_side=512)

    def test_formats_bucket_batches_and_distributed_resume_order(self):
        with tempfile.TemporaryDirectory() as directory:
            c = fixture(Path(directory))
            prepare_data(c, emit=lambda _: None)
            records = read_manifest(c.data.manifest)
            self.assertEqual(len(records), 8)
            d = TrainingDataset(records, c.stages[0], c.data, c.seed)
            d.stage.samples_per_epoch = 128
            d.length = 128
            all_batches = list(AspectRatioBatchSampler(d, 4))
            ranks = [list(AspectRatioBatchSampler(d, 2, 2, rank)) for rank in range(2)]
            self.assertEqual(all_batches, [a + b for a, b in zip(*ranks)])
            self.assertEqual(all_batches, list(AspectRatioBatchSampler(d, 4)))
            self.assertEqual(all_batches[3:], list(RemainingBatches(AspectRatioBatchSampler(d, 4), 3)))
            self.assertEqual(len({key[0] for batch in all_batches for key in batch}), 128)
            self.assertEqual({batch[0][2] for batch in all_batches}, {0, 1, 2, 3})
            for batch in all_batches:
                self.assertEqual(len({key[2] for key in batch}), 1)
                for _, index, bucket in batch:
                    record = d.records[index]
                    self.assertEqual(bucket, nearest_bucket(record["height"], record["width"], d.stage.crop_buckets))
            first = all_batches[0][0]
            self.assertTrue(torch.equal(d[first], d[first]))
            loader = DataLoader(d, batch_sampler=AspectRatioBatchSampler(d, 2), num_workers=2)
            for batch in loader:
                self.assertEqual(batch.shape[:3], (2, 1, 3))
                self.assertIn(list(batch.shape[-2:]), d.stage.crop_buckets)
            d.epoch = 1
            self.assertNotEqual(all_batches, list(AspectRatioBatchSampler(d, 4)))

    def test_orientation_full_image_resize_and_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c = fixture(root)
            exif = Image.Exif()
            exif[274] = 6
            Image.new("RGB", (80, 40), "red").save(root / "train/nested/rotated.jpg", exif=exif)
            prepare_data(c, emit=lambda _: None)
            records = read_manifest(c.data.manifest)
            rotated = next(r for r in records if "rotated" in r["source"])
            self.assertEqual((rotated["height"], rotated["width"]), (80, 40))
            self.assertEqual(load_frame(rotated, 0).shape, (3, 80, 40))
            record = next(r for r in records if r["split"] == "validation" and r["width"] == 71)
            target = validation_frames(record, c.stages[0], c)
            self.assertEqual(target.shape, (1, 3, 35, 71))
            padded = pad_for_codec(target)
            self.assertEqual(padded.shape, (1, 3, 48, 80))
            self.assertTrue(torch.equal(target, padded[..., :35, :71]))
            c.validation.image_max_side = 50
            self.assertEqual(validation_frames(record, c.stages[0], c).shape, (1, 3, 25, 50))

    def test_pooled_metrics_weight_pixels(self):
        results = [dict(qp=0, pixel_count=100, mse=.01, psnr=20, actual_bpp=1),
                   dict(qp=0, pixel_count=300, mse=.001, psnr=30, actual_bpp=2)]
        value = summarize_qps(results)["0"]
        self.assertEqual(value["mean_psnr"], 25)
        self.assertEqual(value["actual_bpp"], 1.75)
        self.assertAlmostEqual(value["pooled_psnr"], -10 * math.log10(.00325))

    def test_bucket_training_resumes_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c = fixture(root)
            prepare_data(c, emit=lambda _: None)
            captured = root / "intermediate.pt"
            original = Trainer.save
            def capture(trainer, stage, filename="last.pt"):
                original(trainer, stage, filename)
                if trainer.global_step == 1 and filename == "last.pt" and not captured.exists():
                    shutil.copyfile(trainer.root / filename, captured)
            with contextlib.redirect_stdout(io.StringIO()), patch.object(Trainer, "save", capture):
                self.assertEqual(train(c), 0)
            first = torch.load(root / "run/last.pt", weights_only=True)
            resumed = copy.deepcopy(c)
            resumed.output.root = str(root / "resumed")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(train(resumed, str(captured)), 0)
            second = torch.load(root / "resumed/last.pt", weights_only=True)
            self.assertEqual(first["global_step"], second["global_step"])
            for key, tensor in first["state_dict"].items():
                self.assertTrue(torch.equal(tensor, second["state_dict"][key]), key)
            old = c.to_dict()
            old["stages"][0]["crop_buckets"] = None
            current = copy.deepcopy(old)
            del old["stages"][0]["crop_buckets"]
            del old["validation"]["image_max_side"]
            self.assertEqual(_comparable_config(old), _comparable_config(current))

    def test_full_image_native_validation_uses_unpadded_pixels(self):
        from src.layers.int16_inference import CUSTOMIZED_INT16_CUDA_INFERENCE
        if not torch.cuda.is_available() or not CUSTOMIZED_INT16_CUDA_INFERENCE:
            self.skipTest("native CUDA INT16 extension required")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c = fixture(root)
            prepare_data(c, emit=lambda _: None)
            c.device = "cuda"
            c.validation.actual_bitstreams = True
            c.validation.max_samples = 4
            path = root / "image.pt"
            atomic_save(path, model_checkpoint(create_model("image", c.model.image), "int16"))
            result = run_isolated(c, c.stages[0], path, None, torch.device("cuda:0"), root / "validation")
            self.assertTrue(result["qat_parity"] and result["round_trip_exact"])
            self.assertEqual(len(result["results"]), 4)
            for row in result["results"]:
                self.assertEqual(row["pixel_count"], row["width"] * row["height"])
                self.assertAlmostEqual(row["actual_bpp"], 8 * Path(row["bitstream"]).stat().st_size / row["pixel_count"])

    def test_two_rank_bucket_training_keeps_models_synchronized(self):
        if not torch.distributed.is_available():
            self.skipTest("distributed PyTorch required")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c = fixture(root)
            prepare_data(c, emit=lambda _: None)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(c.to_dict()))
            script = root / "worker.py"
            script.write_text('''import hashlib, sys
from src.training.config import load_config
from src.training.trainer import Trainer
from src.training.data import atomic_json
class CheckedTrainer(Trainer):
    def log(self, event, **values):
        super().log(event, **values)
        if event == "training_completed":
            h = hashlib.sha256()
            for key, tensor in self.models["image"].state_dict().items():
                h.update(key.encode())
                h.update(tensor.detach().cpu().numpy().tobytes())
            atomic_json(self.root / ("rank-%d.json" % self.rank), h.hexdigest())
raise SystemExit(CheckedTrainer(load_config(sys.argv[1])).run())
''')
            project = str(Path(__file__).resolve().parents[1])
            result = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                                     "--nproc_per_node=2", str(script), str(config_path)],
                                    cwd=project, env=dict(os.environ, PYTHONPATH=project, OMP_NUM_THREADS="1"),
                                    capture_output=True, text=True, timeout=110)
            self.assertEqual(result.returncode, 0, result.stdout[-2000:] + result.stderr[-5000:])
            self.assertEqual((root / "run/rank-0.json").read_text(), (root / "run/rank-1.json").read_text())
