from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class WorkerDevicePlan:
    """Independent worker-to-device assignments for file-level parallelism."""

    worker_count: int
    cuda_indices: Tuple[Optional[int], ...]
    using_cuda: bool

    def describe(self) -> str:
        if not self.using_cuda:
            noun = "worker" if self.worker_count == 1 else "workers"
            return f"{self.worker_count} CPU {noun}"
        assignments = ", ".join(
            f"worker {worker_id}->cuda:{device_index}"
            for worker_id, device_index in enumerate(self.cuda_indices)
        )
        return assignments


def build_worker_device_plan(
    *,
    use_cuda: bool,
    requested_workers: Optional[int],
    requested_cuda_indices: Optional[Sequence[int]],
    cuda_available: bool,
    cuda_device_count: int,
) -> WorkerDevicePlan:
    """Build a validated plan, defaulting to one worker per visible/selected GPU."""

    if requested_workers is not None and requested_workers < 1:
        raise ValueError("worker count must be at least 1")

    requested_devices = list(requested_cuda_indices or [])
    if not use_cuda:
        if requested_devices:
            raise ValueError("CUDA device indices cannot be used when CUDA is disabled")
        worker_count = requested_workers or 1
        return WorkerDevicePlan(worker_count, (None,) * worker_count, False)

    if not cuda_available or cuda_device_count < 1:
        if requested_devices:
            raise ValueError("CUDA device indices were provided, but no CUDA devices are available")
        worker_count = requested_workers or 1
        return WorkerDevicePlan(worker_count, (None,) * worker_count, False)

    selected_devices = requested_devices or list(range(cuda_device_count))
    if len(set(selected_devices)) != len(selected_devices):
        raise ValueError("CUDA device indices must not contain duplicates")
    invalid_devices = [
        index for index in selected_devices
        if index < 0 or index >= cuda_device_count
    ]
    if invalid_devices:
        visible_range = f"0..{cuda_device_count - 1}"
        raise ValueError(
            f"invalid CUDA device index/indices {invalid_devices}; visible logical indices are {visible_range}"
        )

    worker_count = requested_workers or len(selected_devices)
    assignments = tuple(
        selected_devices[worker_id % len(selected_devices)]
        for worker_id in range(worker_count)
    )
    return WorkerDevicePlan(worker_count, assignments, True)
