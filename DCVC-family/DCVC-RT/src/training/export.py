"""Export an unfused model or I/P pair with bound prepared integer state."""

from pathlib import Path
import shutil
import tempfile

import torch

from src.models.architecture import create_model, model_from_metadata
from src.training.checkpoint import atomic_save, model_checkpoint
from src.training.data import atomic_json
from src.utils.common import (architecture_sha256, checkpoint_sha256, checkpoint_state_dict,
                              save_int16_prep_state)


def _write_export(checkpoint_path, output):
    torch.set_num_threads(1)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") not in {"dcvc-rt-model", "dcvc-rt-training"}:
        raise ValueError("export-model requires a training checkpoint or metadata-bearing model checkpoint")
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "manifest.json").exists():
        raise ValueError(f"export destination already contains a model bundle: {destination}")
    if "models" in checkpoint:
        state_dicts = checkpoint["models"]
        models = {kind: create_model(kind, checkpoint["config"]["model"][kind]) for kind in state_dicts}
    else:
        metadata = checkpoint["architecture"]
        kind = metadata["kind"]
        state_dicts = {kind: checkpoint_state_dict(checkpoint)}
        models = {kind: model_from_metadata(metadata, kind)}
    manifest = {"version": 1, "runtime": "int16", "models": {},
                "source_checkpoint_sha256": checkpoint_sha256(checkpoint_path)}
    image_sha = None
    for kind in ("image", "video"):
        if kind not in models:
            continue
        model = models[kind].eval().float()
        model.load_state_dict(state_dicts[kind], strict=True)
        path = destination / f"{kind}.pth.tar"
        if path.exists() or path.with_name(path.name + ".int16prep.pt").exists():
            raise ValueError(f"export would overwrite an existing checkpoint: {path}")
        mode = checkpoint.get("model_modes", {}).get(kind, checkpoint.get("training_mode", "float"))
        payload = model_checkpoint(model, mode,
                                   paired_image_sha256=image_sha if kind == "video" else None)
        atomic_save(path, payload)
        sha = checkpoint_sha256(path)
        if kind == "image":
            image_sha = sha
        # Prepare on CPU exactly as the inference loader does. The copied model
        # may be fused here; the exported trainable checkpoint remains unfused.
        with torch.no_grad():
            model.prepare_int16_inference()
            model.update(checkpoint.get("config", {}).get("quantization", {}).get("force_zero_thres"))
            prepared = model.export_int16_prep()
        prepared.update(version=4, binding={"checkpoint_sha256": sha,
                                           "architecture_sha256": architecture_sha256(payload["architecture"])})
        save_int16_prep_state(path, prepared)
        prepared_path = path.with_name(path.name + ".int16prep.pt")
        manifest["models"][kind] = {"checkpoint": path.name, "sha256": sha,
                                     "architecture": payload["architecture"],
                                     "prepared": prepared_path.name, "prepared_sha256": checkpoint_sha256(prepared_path)}
    atomic_json(destination / "manifest.json", manifest)
    return manifest


def export_checkpoint(checkpoint_path, output):
    destination = Path(output).resolve()
    if destination.exists():
        raise ValueError(f"choose a new export directory; destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".dcvc-export-", dir=destination.parent))
    try:
        manifest = _write_export(checkpoint_path, temporary)
        temporary.rename(destination)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
