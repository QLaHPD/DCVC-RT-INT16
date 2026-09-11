import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from src.layers import int16_inference as integer
from src.layers.layers import DepthConvBlock, SubpelConv2x
from src.models.architecture import ImageArchitecture, VideoArchitecture, create_model
from src.training.checkpoint import atomic_save, model_checkpoint
from src.training.models import TrainingModel, rd_loss
from src.training.ops import TrainingOps
from src.utils.common import load_model_for_inference, load_int16_prep_state


NATIVE = integer.CUSTOMIZED_INT16_CUDA_INFERENCE and torch.cuda.is_available()


def small(kind):
    cls = ImageArchitecture if kind == "image" else VideoArchitecture
    return {key: 8 if "channels" in key else 1 for key in cls().to_dict()}


@unittest.skipUnless(NATIVE, "requires CUDA and the deployed INT16 extension")
class ExactQATTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(8)
        self.environment = patch.dict(os.environ, {"DCVC_USE_INT16": "1", "DCVC_INT16_CUDA_GRAPH": "0"})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_convolution_matches_int64_reference_with_int32_wrap(self):
        op = TrainingOps("int16")
        conv = nn.Conv2d(5, 4, 1).cuda()
        with torch.no_grad():
            conv.weight.fill_(3.75)
            conv.bias.copy_(torch.tensor([1., -1., 62., -62.], device="cuda"))
        x = torch.full((2, 5, 2, 3), 60.0, device="cuda", requires_grad=True)
        exact = op.conv(x, conv)
        raw_x = integer.feature_to_int16(x.detach()).long()
        raw_w = integer.weight_to_int16(conv.weight.detach()).long()
        sums = torch.einsum("bchw,oc->bohw", raw_x.cpu(), raw_w[:, :, 0, 0].cpu())
        sums = (sums + 2**31) % 2**32 - 2**31
        rounded = sums.sign() * ((sums.abs() + 4096) // 8192)
        bias = integer.bias_to_int16(conv.bias.detach()).cpu().long()[None, :, None, None]
        expected = (rounded + bias).clamp(-32768, 32767).float() / 512
        self.assertTrue(torch.equal(exact.cpu(), expected))
        exact.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_blocks_and_fused_postoperations_match_runtime(self):
        op = TrainingOps("int16")
        x = (torch.rand(2, 8, 3, 5, device="cuda") * 120 - 60).requires_grad_()
        x = op.finish(x, op.native(x))
        for block in (DepthConvBlock(8, 8, shortcut=True), SubpelConv2x(8, 8, 1)):
            block = block.cuda()
            actual = op.apply(block, x)
            expected = block(op.native(x))
            self.assertTrue(torch.equal(actual, integer.feature_to_float(expected)))
        conv = nn.Conv2d(8, 8, 1).cuda()
        with torch.no_grad():
            conv.bias.fill_(50)
        scale = torch.full((1, 8, 1, 1), 0.75, device="cuda", requires_grad=True)
        actual = op.conv_post(x, conv, quant=scale)
        expected = integer.conv2d_bias_quant_module_int16(op.native(x), conv, op.native(scale))
        self.assertTrue(torch.equal(actual, integer.feature_to_float(expected)))
        actual.sum().backward()
        self.assertIsNotNone(scale.grad)

    def test_weight_changes_are_not_hidden_by_inference_caches(self):
        op = TrainingOps("int16")
        conv = nn.Conv2d(2, 2, 1).cuda()
        with torch.no_grad():
            conv.weight.fill_(0.01)
            conv.bias.zero_()
        x = torch.ones(1, 2, 2, 2, device="cuda")
        before = op.conv(x, conv)
        with torch.no_grad():
            conv.weight.add_(0.1)
        after = op.conv(x, conv)
        self.assertFalse(torch.equal(before, after))
        self.assertFalse(hasattr(conv, "_int16_cache"))

    def pair(self, kind):
        model = create_model(kind, small(kind))
        native = copy.deepcopy(model).eval()
        native.prepare_int16_inference()
        native.update()
        native.cuda()
        return TrainingModel(model.cuda(), "int16"), native

    def test_image_and_temporal_reference_parity_with_backward_and_refresh(self):
        trained_i, native_i = self.pair("image")
        x = torch.rand(1, 3, 32, 48, device="cuda")
        image = trained_i(x, 63)
        loss, _ = rd_loss(x, image, 63, "image")
        loss.backward()
        self.assertTrue(torch.isfinite(trained_i.codec.q_scale_enc.grad).all())
        with torch.no_grad():
            encoded = native_i.compress(x.half(), 63)
        self.assertTrue(torch.equal(image["x_hat"], encoded["x_hat"]))
        trained_p, native_p = self.pair("video")
        reference = {"x_hat": image["x_hat"].detach(), "feature": None}
        native_p.add_ref_frame(None, encoded["x_hat"])
        for index in range(1, 10):
            qp, refresh = 71 if index % 2 else 0, index % 3 == 0
            result = trained_p(x, qp, reference, refresh=refresh)
            loss, _ = rd_loss(x, result, min(qp, 63), "video")
            loss.backward()
            self.assertTrue(torch.isfinite(trained_p.codec.encoder.conv1.weight.grad).all())
            trained_p.zero_grad(set_to_none=True)
            with torch.no_grad():
                if refresh:
                    native_p.prepare_feature_adaptor_i(last_qp)
                native_p.compress(x.half(), qp)
            self.assertTrue(torch.equal(result["feature"], integer.feature_to_float(native_p.dpb[0].feature)))
            reference = {"x_hat": result["x_hat"].detach(), "feature": result["feature"].detach()}
            last_qp = qp
        self.assertFalse(trained_p.codec.encoder.fuse_conv1_flag)

    def test_metadata_checkpoint_rebuilds_stale_prepared_state(self):
        model = create_model("image", small("image"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.pt"
            atomic_save(path, model_checkpoint(model, "int16"))
            first = load_model_for_inference(type(model), path, "cuda")
            binding_before = load_int16_prep_state(path)["binding"]
            del first
            with torch.no_grad():
                model.enc.enc_1.dc[0].weight.add_(0.01)
            atomic_save(path, model_checkpoint(model, "int16"))
            second = load_model_for_inference(type(model), path, "cuda")
            binding_after = load_int16_prep_state(path)["binding"]
            self.assertNotEqual(binding_before, binding_after)
            expected = integer.weight_to_int16(model.enc.enc_1.dc[0].weight)
            actual, _ = integer.get_quantized_conv_params(second.enc.enc_1.dc[0], torch.device("cuda:0"))
            self.assertTrue(torch.equal(expected, actual.cpu()))


if __name__ == "__main__":
    unittest.main()
