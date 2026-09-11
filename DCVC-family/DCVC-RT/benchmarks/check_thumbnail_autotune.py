#!/usr/bin/env python3
"""Check first-image-only autotuning and exact streams across thumbnail sizes."""
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ['DCVC_USE_INT16'] = '1'
os.environ['DCVC_INT16_AUTOTUNE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch

from src.cli.thumbnail_codec import discover_thumbnails, encode_thumbnail
from src.layers import int16_inference as integer
from src.models.image_model import DMCI
from src.utils.common import load_model_for_inference, set_torch_env


def main():
    set_torch_env()
    model = load_model_for_inference(DMCI(), str(ROOT/'checkpoints/cvpr2025_image.pth.tar'),
                                    torch.device('cuda:0'))
    rng = np.random.default_rng(17)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for index, (width, height) in enumerate(((176, 96), (128, 72), (33, 19))):
            Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8)).save(
                root/f'{index}.png')
        tasks = discover_thumbnails('synthetic', root, root, qp=45).pending_tasks
        integer._RESIDUAL_TILES.clear()
        sizes, streams = [], []
        for task in tasks:
            encode_thumbnail(task, model)
            sizes.append(len(integer._RESIDUAL_TILES))
            streams.append(Path(task.output_path).read_bytes())
        assert sizes[0] > 0 and sizes == [sizes[0]] * len(tasks), sizes
        # Recreate the previous policy: permit new shapes for every image.
        for task, expected in zip(tasks, streams):
            model._thumbnail_autotune_started = False
            encode_thumbnail(task, model)
            assert Path(task.output_path).read_bytes() == expected
        assert len(integer._RESIDUAL_TILES) > sizes[0]
        print(json.dumps(dict(first_only_cache_sizes=sizes,
                              unrestricted_cache_size=len(integer._RESIDUAL_TILES),
                              exact_images=len(tasks))))


if __name__ == '__main__':
    main()
