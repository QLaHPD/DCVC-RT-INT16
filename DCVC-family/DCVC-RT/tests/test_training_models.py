import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.models.architecture import (ImageArchitecture, VideoArchitecture,
                                     architecture_metadata, create_model)
from src.training.config import TrainingConfig, load_config
from src.training.models import TrainingModel, rd_loss
from src.training.ops import ExactValue
from src.training.checkpoint import atomic_save, model_checkpoint
from src.training.export import export_checkpoint
from src.utils.common import load_model_for_inference


def tiny_config(kind):
    cls = ImageArchitecture if kind == "image" else VideoArchitecture
    return {key: 8 if "channels" in key else 1 for key in cls().to_dict()}


class ArchitectureTests(unittest.TestCase):
    def test_default_checkpoint_schema_is_unchanged(self):
        # Fingerprints captured from unmodified 0833356 checkpoint keys/shapes.
        fingerprints = {"image": "34105c4d7228e7e9d4142660dd37873401f36a91a49dce357ba7337be2190eb1",
                        "video": "d10869194ab9cf2106d87df516efe49eb3cb34ea02597340f5a52383f28edbe3"}
        for kind, expected in fingerprints.items():
            with torch.device("meta"):
                model = create_model(kind)
            shapes = [(key, list(value.shape)) for key, value in model.state_dict().items()]
            self.assertEqual(hashlib.sha256(json.dumps(shapes).encode()).hexdigest(), expected)

    def test_models_do_not_share_configuration(self):
        small = create_model("image", tiny_config("image"))
        with torch.device("meta"):
            default = create_model("image")
            larger = create_model("video", {"channels": 384, "recon_channels": 480,
                                            "latent_channels": 192, "hyper_channels": 192,
                                            "encoder_blocks": 4, "decoder_blocks": 5})
        self.assertEqual(small.enc.enc_1.dc[0].in_channels, 8)
        self.assertEqual(default.enc.enc_1.dc[0].in_channels, 368)
        self.assertEqual(len(larger.encoder.conv2), 4)
        self.assertEqual(larger.q_recon.shape, (72, 480, 1, 1))

    def test_invalid_architecture_is_rejected(self):
        for config in ({"latent_channels": 7}, {"channels": True}, {"decoder_blocks": 0}, {"typo": 2}):
            with self.assertRaises(ValueError):
                ImageArchitecture.from_dict(config)

    def test_inference_loads_custom_architecture_without_default_allocation(self):
        model = create_model("image", tiny_config("image"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "small.pt"
            torch.save({"architecture": architecture_metadata(model), "state_dict": model.state_dict()}, path)
            with patch.object(type(model), "update"):
                loaded = load_model_for_inference(type(model), path, "cpu")
            self.assertEqual(loaded.architecture, model.architecture)
            for key, tensor in model.state_dict().items():
                self.assertTrue(torch.equal(tensor, loaded.state_dict()[key]))


class ConfigTests(unittest.TestCase):
    def recipe(self):
        return {"version": 1, "task": "image", "device": "cpu",
                "validation": {"actual_bitstreams": False},
                "stages": [{"name": "warmup", "model": "image", "mode": "float", "epochs": 1, "lr": 0.001}]}

    def test_json_and_yaml_resolve_paths_against_config(self):
        import yaml
        with tempfile.TemporaryDirectory() as directory:
            for suffix in ("yaml", "json"):
                path = Path(directory) / f"train.{suffix}"
                raw = self.recipe()
                path.write_text(yaml.safe_dump(raw) if suffix == "yaml" else json.dumps(raw))
                cfg = load_config(path)
                self.assertEqual(cfg.output.root, str(Path(directory) / "runs/training"))
                self.assertEqual(cfg.model.image["channels"], 368)

    def test_unknown_fields_and_invalid_modes_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown"):
            TrainingConfig(**{**self.recipe(), "optimizer": {"learnig_rate": 1}})
        raw = self.recipe()
        raw["stages"][0]["mode"] = "int16"
        with self.assertRaisesRegex(ValueError, "CUDA"):
            TrainingConfig(**raw)


class DifferentiableModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(11)

    def assert_parameter_updates(self, model, prefixes, x, qp, **kwargs):
        before = {name: value.detach().clone() for name, value in model.codec.named_parameters()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        result = model(x, qp, **kwargs)
        loss, _ = rd_loss(x, result, min(qp, 63), model.kind)
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        optimizer.step()
        changed = [name for name, p in model.codec.named_parameters() if not torch.equal(before[name], p)]
        for prefix in prefixes:
            self.assertTrue(any(name.startswith(prefix) for name in changed), (prefix, changed))
        return result

    def test_image_encoder_decoder_priors_and_quality_banks_train(self):
        model = TrainingModel(create_model("image", tiny_config("image")))
        x = torch.rand(2, 3, 32, 48)
        self.assert_parameter_updates(model, ["enc.", "dec.", "hyper_enc.", "hyper_dec.",
                                             "bit_estimator_z.", "y_prior_fusion.", "q_scale_"], x, 25)

    def test_video_has_temporal_gradients_and_refresh_path(self):
        model = TrainingModel(create_model("video", tiny_config("video")))
        x = torch.rand(1, 3, 32, 48)
        reference = {"x_hat": x * 0.9, "feature": None}
        first = model(x, 35, reference)
        first["feature"].retain_grad()
        second = model(x * 0.95, 31, first)
        loss, _ = rd_loss(x, second, 31, "video")
        loss.backward()
        self.assertIsNotNone(first["feature"].grad)
        self.assertGreater(first["feature"].grad.abs().sum().item(), 0)
        model.zero_grad(set_to_none=True)
        reference = {"x_hat": first["x_hat"].detach(), "feature": first["feature"].detach()}
        self.assert_parameter_updates(model, ["encoder.", "decoder.", "recon_generation_net.",
                                             "feature_adaptor_i.", "bit_estimator_z.", "q_recon"],
                                      x, 36, reference=reference, refresh=True)

    def test_float_training_ignores_inference_environment(self):
        with patch.dict(os.environ, {"DCVC_USE_INT16": "1", "DCVC_INT16_CUDA_GRAPH": "1"}):
            model = TrainingModel(create_model("image", tiny_config("image")))
            loss, _ = rd_loss(torch.ones(1, 3, 32, 32), model(torch.ones(1, 3, 32, 32), 5), 5, "image")
            loss.backward()
            self.assertTrue(torch.isfinite(loss))

    def test_exact_value_does_not_lose_precision_to_cancellation(self):
        surrogate = torch.tensor([1e20], requires_grad=True)
        out = ExactValue.apply(surrogate, torch.tensor([1 / 512]))
        self.assertEqual(out.item(), 1 / 512)
        out.sum().backward()
        self.assertEqual(surrogate.grad.item(), 1)

    def test_activation_checkpointing_preserves_loss_and_gradients(self):
        import copy
        codec = create_model("image", tiny_config("image"))
        normal = TrainingModel(codec)
        recomputed = TrainingModel(copy.deepcopy(codec), activation_checkpointing=True)
        x = torch.rand(2, 3, 32, 48)
        results = []
        for model in (normal, recomputed):
            torch.manual_seed(31)
            loss, _ = rd_loss(x, model(x, 36), 36, "image")
            loss.backward()
            results.append(loss.detach())
        self.assertTrue(torch.equal(*results))
        for (name, first), (_, second) in zip(normal.named_parameters(), recomputed.named_parameters()):
            if first.grad is None:
                self.assertIsNone(second.grad, name)
            else:
                self.assertTrue(torch.equal(first.grad, second.grad), name)


class ExportTransactionTests(unittest.TestCase):
    def test_failure_does_not_publish_partial_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = create_model("image", tiny_config("image"))
            atomic_save(root / "source.pt", model_checkpoint(model, "int16"))
            with patch.object(type(model), "update"), \
                    patch.object(type(model), "export_int16_prep", return_value={"version": 3}), \
                    patch("src.training.export.save_int16_prep_state", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    export_checkpoint(root / "source.pt", root / "export")
            self.assertFalse((root / "export").exists())
            self.assertEqual(list(root.glob(".dcvc-export-*")), [])
            self.assertTrue((root / "source.pt").is_file())

    def test_existing_export_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "important.txt").write_text("keep")
            with self.assertRaisesRegex(ValueError, "already exists"):
                export_checkpoint("unused.pt", root)
            self.assertEqual((root / "important.txt").read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
