#!/usr/bin/env python3
"""Profile INT16 convolution shapes for one I, reset-P, and steady-P frame."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.image_model import DMCI
from src.models.video_model import DMC
from src.utils.common import load_model_for_inference, set_torch_env
import src.layers.int16_inference as int16_ops


def make_synthetic_frames(device: torch.device):
    """Return deterministic, redistribution-safe YCbCr-like model inputs."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0xDC0C2025)
    frames = torch.randint(
        0, 256, (3, 1, 3, 96, 176), dtype=torch.int16, generator=generator
    )
    return [frame.to(device=device, dtype=torch.float16) / 255.0 for frame in frames]


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    set_torch_env()
    device = torch.device("cuda:0")
    phase = {"name": "load"}
    records = []
    original_conv = int16_ops.conv2d_int16_cuda

    def profiled_conv(x, weight, bias, stride_h, stride_w, pad_h, pad_w, groups):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = original_conv(
            x, weight, bias, stride_h, stride_w, pad_h, pad_w, groups
        )
        end.record()
        records.append((
            phase["name"], tuple(x.shape), tuple(weight.shape), bias is not None,
            stride_h, stride_w, pad_h, pad_w, groups, start, end,
        ))
        return output

    int16_ops.conv2d_int16_cuda = profiled_conv

    image_model = load_model_for_inference(
        DMCI(), str(REPO_ROOT / "checkpoints/cvpr2025_image.pth.tar"), device, None
    )
    video_model = load_model_for_inference(
        DMC(), str(REPO_ROOT / "checkpoints/cvpr2025_video.pth.tar"), device, None
    )
    image_model.set_use_two_entropy_coders(False)
    video_model.set_use_two_entropy_coders(False)
    video_model.set_curr_poc(0)

    frames = make_synthetic_frames(device)

    phase["name"] = "I"
    encoded = image_model.compress(frames[0], 35)
    video_model.clear_dpb()
    video_model.add_ref_frame(None, encoded["x_hat"])

    phase["name"] = "P-reset"
    video_model.prepare_feature_adaptor_i(0)
    qp = video_model.shift_qp(14, 1)
    video_model.compress(frames[1], qp)

    phase["name"] = "P-steady"
    qp = video_model.shift_qp(14, 0)
    video_model.compress(frames[2], qp)
    torch.cuda.synchronize()

    aggregates = defaultdict(lambda: [0, 0.0, 0])
    phase_totals = defaultdict(lambda: [0, 0.0, 0])
    for item in records:
        phase_name, x_shape, weight_shape, has_bias, sh, sw, ph, pw, groups, start, end = item
        elapsed = start.elapsed_time(end)
        out_h = (x_shape[2] + 2 * ph - weight_shape[2]) // sh + 1
        out_w = (x_shape[3] + 2 * pw - weight_shape[3]) // sw + 1
        macs = (
            x_shape[0] * weight_shape[0] * out_h * out_w * weight_shape[1]
            * weight_shape[2] * weight_shape[3]
        )
        key = (
            phase_name, x_shape[1], weight_shape[0], x_shape[2], x_shape[3],
            weight_shape[2], weight_shape[3], sh, sw, ph, pw, groups, has_bias,
        )
        aggregates[key][0] += 1
        aggregates[key][1] += elapsed
        aggregates[key][2] += macs
        phase_totals[phase_name][0] += 1
        phase_totals[phase_name][1] += elapsed
        phase_totals[phase_name][2] += macs

    print(f"extension={int16_ops._INT16_EXT.__file__}")
    for phase_name in ("I", "P-reset", "P-steady"):
        count, elapsed, macs = phase_totals[phase_name]
        print(
            f"phase={phase_name} convs={count} cuda_ms={elapsed:.3f} "
            f"gmac={macs / 1.0e9:.4f}"
        )

    print("phase count total_ms inC outC H W kH kW stride pad groups bias")
    for key, values in sorted(aggregates.items(), key=lambda item: item[1][1], reverse=True):
        phase_name, in_c, out_c, height, width, kh, kw, sh, sw, ph, pw, groups, has_bias = key
        count, elapsed, _ = values
        print(
            f"{phase_name} {count} {elapsed:.4f} {in_c} {out_c} {height} {width} "
            f"{kh} {kw} {sh}x{sw} {ph}x{pw} {groups} {int(has_bias)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
