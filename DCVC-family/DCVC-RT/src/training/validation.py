"""Isolated deployment validation; also checks QAT/export arithmetic parity."""

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import torch

from src.codec.frame_decoder import BitstreamFrameSource, DecoderModels
from src.models.architecture import create_model
from src.models.image_model import DMCI
from src.models.video_model import DMC
from src.training.checkpoint import load_initial_weights
from src.training.config import TrainingConfig, StageConfig
from src.training.data import TrainingDataset, atomic_json, read_manifest
from src.training.models import QP_OFFSETS, TrainingModel, distortion
from src.training.ops import TrainingOps
from src.utils.common import load_model_for_inference, set_torch_env
from src.utils.stream_helper import write_ip, write_sps


def _shadow(kind, config, path, device):
    model = create_model(kind, getattr(config.model, kind))
    load_initial_weights(model, path)
    return TrainingModel(model.to(device).float(), "int16",
                         force_zero_thres=config.quantization.force_zero_thres).eval()


@torch.no_grad()
def validate_actual(config, stage, image_path, video_path=None, device="cuda:0", artifact_root=None):
    os.environ["DCVC_USE_INT16"] = "1"
    set_torch_env()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("deployment validation requires a CUDA device")
    torch.cuda.set_device(device)
    thres = config.quantization.force_zero_thres
    native_i = load_model_for_inference(DMCI, image_path, device, thres)
    native_p = load_model_for_inference(DMC, video_path, device, thres) if stage.model == "video" else None
    shadow_i = _shadow("image", config, image_path, device)
    shadow_p = _shadow("video", config, video_path, device) if native_p is not None else None
    records = read_manifest(config.data.manifest)
    validation_stage = replace(stage, samples_per_epoch=None)
    # Evaluate as many consecutive frames as each source actually has; a short
    # validation clip remains useful even when training uses a longer stage.
    candidates = [r for r in records if r["split"] == "validation"
                  and (stage.model == "image" or r["kind"] == "video" and len(r["frames"]) >= 2)]
    if not candidates:
        raise ValueError("no validation sources for deployment evaluation")
    results = []
    root = Path(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    for sample_index, record in enumerate(candidates[:config.validation.max_samples]):
        length = 1 if stage.model == "image" else min(len(record["frames"]), config.validation.max_frames)
        sample_stage = replace(validation_stage, sequence_length=length)
        frames = TrainingDataset([record], sample_stage, config.data, config.seed, "validation")[0]
        height, width = frames.shape[-2:]
        for qp in config.validation.qps:
            extension = ".dcvci" if stage.model == "image" else ".bin"
            path = root / f"sample-{sample_index:04d}-q{qp}{extension}"
            expected, feature_states, floating = [], [], []
            estimates = []
            reference, float_reference = None, None
            native_i.set_use_two_entropy_coders(False)
            if native_p is not None:
                native_p.clear_dpb()
                native_p.set_use_two_entropy_coders(False)
            last_qp = qp
            written_sps = set()
            with path.open("wb") as stream:
                for index, frame in enumerate(frames):
                    x = frame.unsqueeze(0).to(device)
                    is_i = index == 0
                    refresh = not is_i and config.validation.reset_interval > 0 \
                        and index % config.validation.reset_interval == 1
                    current_qp = qp if is_i else qp + QP_OFFSETS[index % 8]
                    shadow = shadow_i if is_i else shadow_p
                    shadow.ops = TrainingOps("int16", force_zero_thres=thres)
                    simulated = shadow(x, current_qp, reference, refresh)
                    expected.append(simulated["x_hat"].cpu())
                    estimates.append(float((simulated["bits_y"] + simulated["bits_z"]).sum()))
                    reference = {"x_hat": simulated["x_hat"], "feature": simulated["feature"]}
                    shadow.ops = TrainingOps("float", force_zero_thres=thres)
                    float_result = shadow(x, current_qp, float_reference, refresh)
                    floating.append(float_result["x_hat"].cpu())
                    float_reference = {"x_hat": float_result["x_hat"], "feature": float_result["feature"]}
                    if is_i:
                        encoded = native_i.compress(x.half(), current_qp)
                        if not torch.equal(encoded["x_hat"], simulated["x_hat"]):
                            raise RuntimeError("QAT I-frame forward differs from prepared encoder")
                        if native_p is not None:
                            native_p.add_ref_frame(None, encoded["x_hat"])
                        feature_states.append(None)
                    else:
                        if refresh:
                            native_p.prepare_feature_adaptor_i(last_qp)
                        encoded = native_p.compress(x.half(), current_qp)
                        feature = native_p.dpb[0].feature
                        if not torch.equal(feature.float() / 512, simulated["feature"]):
                            raise RuntimeError(f"QAT P-frame reference differs from prepared encoder at frame {index}")
                        feature_states.append(feature.cpu())
                    last_qp = current_qp
                    # Two reusable SPS IDs encode the feature-refresh flag.
                    sps_id = int(refresh)
                    if sps_id not in written_sps:
                        write_sps(stream, {"sps_id": sps_id, "height": height, "width": width,
                                           "ec_part": 0, "use_ada_i": int(refresh)})
                        written_sps.add(sps_id)
                    write_ip(stream, is_i, sps_id, current_qp, encoded["bit_stream"])
            reconstructed = []
            decoder_models = DecoderModels(native_i, native_p, device)
            with BitstreamFrameSource(path, decoder_models) as source:
                for index, decoded in enumerate(source.iter_frames(output_format="tensor")):
                    image = decoded.tensor.cpu()
                    if not torch.equal(image, expected[index]):
                        raise RuntimeError(f"exported bitstream reconstruction mismatch at frame {index}")
                    if index and not torch.equal(native_p.dpb[0].feature.cpu(), feature_states[index]):
                        raise RuntimeError(f"exported bitstream reference drift at frame {index}")
                    reconstructed.append(image)
            if len(reconstructed) != length:
                raise RuntimeError("deployment validation decoded the wrong number of frames")
            target, reconstructed = frames, torch.cat(reconstructed, dim=0)
            mse = float((target - reconstructed).square().mean())
            d = float(distortion(target, reconstructed, config.loss).mean())
            actual_bpp = 8 * path.stat().st_size / (height * width * length)
            endpoints = config.loss.image_lambdas if stage.model == "image" else config.loss.video_lambdas
            lam = math.exp(math.log(endpoints[0]) + qp / 63 * math.log(endpoints[1] / endpoints[0]))
            results.append({"source_id": record["id"], "qp": qp, "frames": length,
                            "actual_bpp": actual_bpp, "estimated_bpp": sum(estimates) / (height * width * length),
                            "psnr": -10 * math.log10(max(mse, 1e-12)), "distortion": d,
                            "float_to_int16_mse": float((torch.cat(floating) - reconstructed).square().mean()),
                            "rd": actual_bpp + lam * d, "bitstream": str(path)})
    return {"kind": "actual_int16", "score": sum(r["rd"] for r in results) / len(results),
            "qat_parity": True, "round_trip_exact": True, "results": results}


@torch.no_grad()
def validate_estimated(primary, image, config, stage, records, device):
    """Fast float-only smoke validation; never presented as coded bitrate."""
    from src.training.sequence import TrainingWindow, temporal_windows
    window = TrainingWindow(primary, stage, config, image).eval()
    dataset = TrainingDataset(records, stage, config.data, config.seed, "validation")
    values = []
    for index in range(min(len(dataset), config.validation.max_samples)):
        frames = dataset[index].unsqueeze(0).to(device)
        for qp in config.validation.qps:
            reference, score = None, 0.0
            for chunk, start, weight in temporal_windows(frames, stage):
                loss, _, reference = window(chunk, qp, start, reference)
                score += float(loss) * weight
            values.append(score)
    return {"kind": "estimated", "score": sum(values) / len(values), "samples": len(values)}


def run_isolated(config, stage, image_path, video_path, device, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    job = directory / "job.json"
    result = directory / "result.json"
    atomic_json(job, {"config": config.to_dict(), "stage": stage.__dict__,
                      "image_path": str(image_path), "video_path": str(video_path) if video_path else None,
                      "device": str(device), "artifact_root": str(directory / "bitstreams")})
    env = dict(os.environ, DCVC_USE_INT16="1", DCVC_INT16_AUTOTUNE="0", DCVC_INT16_CUDA_GRAPH="0")
    with (directory / "worker.log").open("w") as log:
        process = subprocess.run([sys.executable, "-m", "src.training.validation", "--job", str(job),
                                  "--result", str(result)], stdout=log, stderr=subprocess.STDOUT, env=env,
                                 cwd=Path(__file__).resolve().parents[2])
    if process.returncode:
        raise RuntimeError(f"deployment validation failed; see {directory / 'worker.log'}")
    return json.loads(result.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    job = json.loads(Path(args.job).read_text())
    config = TrainingConfig(**job.pop("config"))
    stage = StageConfig(**job.pop("stage"))
    atomic_json(args.result, validate_actual(config, stage, **job))


if __name__ == "__main__":
    main()
