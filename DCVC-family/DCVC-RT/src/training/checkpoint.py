"""Atomic checkpoints containing only weights-only-loadable Python values."""

from datetime import datetime, timezone
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from src.models.architecture import architecture_metadata
from src.utils.common import checkpoint_state_dict


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    return value


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(cpu_tree(value), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def rng_state():
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "numpy": (numpy_state[0], torch.from_numpy(numpy_state[1].astype(np.int64)),
                      *numpy_state[2:]),
            "cuda": {"device": torch.cuda.current_device(),
                     "state": torch.cuda.get_rng_state(torch.cuda.current_device())}
            if torch.cuda.is_initialized() else None}


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], numpy_state[1].numpy().astype(np.uint32), *numpy_state[2:]))
    if state["cuda"]:
        torch.cuda.set_rng_state(state["cuda"]["state"], state["cuda"]["device"])


def load_initial_weights(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") is not None and checkpoint["architecture"] != architecture_metadata(model):
        raise ValueError(f"checkpoint architecture differs from configured model: {path}; "
                         "use matching dimensions or train the new architecture from scratch")
    try:
        model.load_state_dict(checkpoint_state_dict(checkpoint), strict=True)
    except RuntimeError as exc:
        raise ValueError(f"checkpoint parameters do not match the configured architecture: {path}") from exc
    return checkpoint


def model_checkpoint(model, mode, *, paired_image_sha256=None):
    return {"format": "dcvc-rt-model", "version": 1,
            "architecture": architecture_metadata(model),
            "training_mode": mode, "state_dict": cpu_tree(model.state_dict()),
            "paired_image_sha256": paired_image_sha256,
            "created_utc": datetime.now(timezone.utc).isoformat()}
