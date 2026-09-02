#!/usr/bin/env python3
"""Print a CUDA operator profile for one production-like steady P-frame."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_int16_conv_shapes import make_synthetic_frames
from src.models.image_model import DMCI
from src.models.video_model import DMC
from src.utils.common import load_model_for_inference, set_torch_env


def main() -> int:
    set_torch_env()
    device = torch.device("cuda:0")
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

    encoded = image_model.compress(frames[0], 35)
    video_model.clear_dpb()
    video_model.add_ref_frame(None, encoded["x_hat"])
    video_model.prepare_feature_adaptor_i(0)
    video_model.compress(frames[1], video_model.shift_qp(14, 1))

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=True) as profile:
        video_model.compress(frames[2], video_model.shift_qp(14, 0))
        torch.cuda.synchronize()

    print(profile.key_averages().table(sort_by="self_cuda_time_total", row_limit=80))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
