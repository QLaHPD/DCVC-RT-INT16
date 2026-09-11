"""A DDP forward covers one temporal gradient window, followed by backward."""

import torch
from torch import nn

from src.training.models import QP_OFFSETS, TrainingModel, rd_loss


class TrainingWindow(nn.Module):
    def __init__(self, primary, stage, config, image=None):
        super().__init__()
        self.primary = TrainingModel(primary, stage.mode, config.optimizer.activation_checkpointing,
                                     config.quantization.force_zero_thres)
        self.image = None
        self.stage, self.config = stage, config
        if stage.model == "video":
            if image is None:
                raise ValueError("video windows require a frozen image model")
            image.requires_grad_(False)
            self.image = TrainingModel(image, stage.mode, False, config.quantization.force_zero_thres).eval()

    def train(self, mode=True):
        super().train(mode)
        if self.image is not None:
            self.image.eval()
        return self

    def forward(self, frames, qp, start_index=0, reference=None):
        total, metrics, count = None, {}, 0
        for offset in range(frames.shape[1]):
            frame_index = start_index + offset
            x = frames[:, offset]
            if self.stage.model == "video" and frame_index == 0:
                with torch.no_grad():
                    result = self.image(x, qp)
                reference = {"x_hat": result["x_hat"], "feature": None}
                continue
            if self.stage.model == "image":
                result = self.primary(x, qp)
            else:
                period = self.config.validation.reset_interval
                refresh = period > 0 and frame_index % period == 1
                result = self.primary(x, qp + QP_OFFSETS[frame_index % 8], reference, refresh)
                reference = {"x_hat": result["x_hat"], "feature": result["feature"]}
            loss, values = rd_loss(x, result, qp, self.stage.model, self.config.loss, frame_index)
            total = loss if total is None else total + loss
            for key, value in values.items():
                metrics[key] = metrics.get(key, 0) + value
            count += 1
        if count == 0:
            raise ValueError("a temporal gradient window must contain a trainable frame")
        # State passed to the next window is detached. Earlier windows can be
        # backpropagated and freed immediately instead of retaining a whole GOP.
        reference = None if reference is None else {
            key: value.detach() if value is not None else None for key, value in reference.items()}
        return total / count, {key: value / count for key, value in metrics.items()}, reference


def temporal_windows(frames, stage):
    if stage.model == "image":
        yield frames, 0, 1.0
        return
    length = frames.shape[1]
    width = stage.temporal_gradient_length
    for start in range(1, length, width):
        end = min(length, start + width)
        offset = 0 if start == 1 else start
        yield frames[:, offset:end], offset, (end - start) / (length - 1)
