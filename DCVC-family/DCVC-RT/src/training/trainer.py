"""Single-node staged training, with optimizer-boundary resume and DDP."""

from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import fcntl
import gc
import itertools
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler, RandomSampler

from src.models.architecture import architecture_metadata, create_model
from src.training.checkpoint import (atomic_save, cpu_tree, load_initial_weights,
                                     model_checkpoint, restore_rng, rng_state)
from src.training.data import TrainingDataset, atomic_json, digest, read_manifest
from src.training.sequence import TrainingWindow, temporal_windows
from src.training.validation import run_isolated, validate_estimated
from src.training.aspect_ratio import AspectRatioBatchSampler
from src.training.config import TrainingConfig


class RemainingBatches:
    def __init__(self, batches, start):
        self.batches, self.start = batches, start

    def __iter__(self):
        return itertools.islice(iter(self.batches), self.start, None)

    def __len__(self):
        return max(0, len(self.batches) - self.start)


def _comparable_config(config):
    # Fill newly introduced optional defaults for older checkpoints too.
    config = TrainingConfig(**config).to_dict()
    # Reporting destinations/frequency may change when resuming. Optimization,
    # source inventory and schedules may not silently change underneath a run.
    config.pop("output", None)
    config.pop("distributed_timeout_seconds", None)
    return config


class Trainer:
    def __init__(self, config, resume=None):
        self.config, self.resume_path = config, resume
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.distributed = self.world_size > 1
        self.device = torch.device(f"cuda:{self.local_rank}" if config.device == "cuda" else "cpu")
        self.root = Path(config.output.root)
        self.models = {}
        self.model_modes = {}
        self.optimizer = None
        self.optimizer_kind = None
        self.window = None
        self.ddp = None
        self.global_step = 0
        self.last_validation_step = 0
        self.position = {"stage_index": 0, "epoch": 0, "batch_index": 0, "stage_step": 0}
        self.best = {}
        self.stop_requested = False
        self.consecutive_skips = 0
        self.lock = None

    def log(self, event, **values):
        if self.rank:
            return
        row = {"time": datetime.now(timezone.utc).isoformat(), "event": event,
               "step": self.global_step, **values}
        def finite(value):
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, dict):
                return {key: finite(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [finite(item) for item in value]
            return value
        row = finite(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        with (self.root / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    def root_call(self, function):
        answer = [None]
        if self.rank == 0:
            try:
                answer[0] = {"value": function()}
            except Exception as exc:
                answer[0] = {"error": f"{type(exc).__name__}: {exc}"}
        if self.distributed:
            dist.broadcast_object_list(answer, src=0, device=self.device)
        if "error" in answer[0]:
            raise RuntimeError(answer[0]["error"])
        return answer[0]["value"]

    def flag_anywhere(self, value):
        flag = torch.tensor(int(value), device=self.device)
        if self.distributed:
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def setup(self):
        torch.set_num_threads(1)
        if self.device.type == "cuda":
            if not torch.cuda.is_available() or self.local_rank >= torch.cuda.device_count():
                raise RuntimeError(f"CUDA GPU {self.local_rank} is unavailable")
            if self.config.validation.actual_bitstreams or any(s.mode == "int16" for s in self.config.stages):
                from src.layers.int16_inference import CUSTOMIZED_INT16_CUDA_INFERENCE
                if not CUSTOMIZED_INT16_CUDA_INFERENCE:
                    raise RuntimeError("INT16 stages/validation require ./build_native_extensions.sh before training")
            torch.cuda.set_device(self.device)
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        if self.distributed:
            if not dist.is_available():
                raise RuntimeError("this PyTorch build has no distributed support; use a DDP-enabled PyTorch build")
            dist.init_process_group("nccl" if self.device.type == "cuda" else "gloo",
                                    timeout=timedelta(seconds=self.config.distributed_timeout_seconds))
        random.seed(self.config.seed + self.rank)
        np.random.seed((self.config.seed + self.rank) % 2**32)
        torch.manual_seed(self.config.seed + self.rank)

        def open_run():
            self.root.mkdir(parents=True, exist_ok=True)
            self.lock = (self.root / ".train.lock").open("a+")
            try:
                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"another trainer owns {self.root}") from exc
            if not self.resume_path and (self.root / "last.pt").exists():
                raise ValueError("run already has a checkpoint; use --resume or a new output.root")
            atomic_json(self.root / "config.resolved.json", self.config.to_dict())
        self.root_call(open_run)
        self.records = read_manifest(self.config.data.manifest)
        self.manifest_hash = digest(self.records)
        # Validate every stage's usable sources before creating any model.
        for stage in self.config.stages:
            TrainingDataset(self.records, stage, self.config.data, self.config.seed)
            if not self.config.validation.actual_bitstreams:
                TrainingDataset(self.records, stage, self.config.data, self.config.seed, "validation")
        self.resume = None
        if self.resume_path:
            self.resume = torch.load(self.resume_path, map_location="cpu", weights_only=True)
            if self.resume.get("format") != "dcvc-rt-training" or self.resume.get("version") != 1:
                raise ValueError("--resume requires a training checkpoint, not an inference export")
            if self.resume["world_size"] != self.world_size:
                raise ValueError("exact resume requires the original GPU/process count")
            if _comparable_config(self.resume["config"]) != _comparable_config(self.config.to_dict()):
                raise ValueError("resume config changes training/data/architecture settings; start a fine-tuning run instead")
            if self.resume["manifest_sha256"] != self.manifest_hash:
                raise ValueError("the source manifest changed since this checkpoint")
            self.position = self.resume["position"]
            self.global_step = self.resume["global_step"]
            self.last_validation_step = self.resume.get("last_validation_step", 0)
            self.best = self.resume["best"]
            self.consecutive_skips = self.resume.get("consecutive_skips", 0)
            self.model_modes = self.resume.get("model_modes", {})
            for kind, state in self.resume["models"].items():
                model = create_model(kind, getattr(self.config.model, kind))
                model.load_state_dict(state, strict=True)
                self.models[kind] = model.to(self.device).float()
        self.log("training_started", world_size=self.world_size, device=str(self.device), resume=self.resume_path)

    def model(self, kind):
        if kind not in self.models:
            model = create_model(kind, getattr(self.config.model, kind))
            initial = getattr(self.config.model, f"init_{kind}")
            if initial:
                metadata = load_initial_weights(model, initial)
                self.model_modes[kind] = metadata.get("training_mode", "float")
            else:
                self.model_modes[kind] = "float"
            self.models[kind] = model.to(self.device).float()
        return self.models[kind]

    def wrap(self):
        self.ddp = DistributedDataParallel(
            self.window, device_ids=[self.local_rank] if self.device.type == "cuda" else None,
            broadcast_buffers=False, find_unused_parameters=True) if self.distributed else self.window

    def enter_stage(self, stage):
        primary = self.model(stage.model)
        self.model_modes[stage.model] = stage.mode
        primary.requires_grad_(True)
        self.ddp = None
        self.window = None
        if self.optimizer_kind != stage.model:
            self.optimizer = None
            gc.collect()
            optimizer = self.config.optimizer
            self.optimizer = torch.optim.AdamW(primary.parameters(), lr=stage.lr, betas=tuple(optimizer.betas),
                                               eps=optimizer.eps, weight_decay=optimizer.weight_decay)
            self.optimizer_kind = stage.model
        image = self.model("image") if stage.model == "video" else None
        self.window = TrainingWindow(primary, stage, self.config, image).train()
        self.wrap()
        if self.resume is not None:
            self.optimizer.load_state_dict(self.resume["optimizer"])
            restore_rng(self.resume["rng_states"][self.rank])
            self.resume = None
        self.optimizer.zero_grad(set_to_none=True)

    def save(self, stage, filename="last.pt"):
        state = rng_state()
        states = [None] * self.world_size if self.rank == 0 else None
        if self.distributed:
            dist.gather_object(state, states, dst=0)
        else:
            states = [state]
        def publish():
            models = {kind: cpu_tree(model.state_dict()) for kind, model in self.models.items()}
            payload = {"format": "dcvc-rt-training", "version": 1,
                       "architecture": architecture_metadata(self.models[stage.model]),
                       "training_mode": stage.mode, "state_dict": models[stage.model], "models": models,
                       "model_modes": self.model_modes,
                       "config": self.config.to_dict(), "optimizer": self.optimizer.state_dict(),
                       "scheduler": {"policy": "cosine_by_epoch", "stage": stage.name},
                       "global_step": self.global_step, "position": dict(self.position),
                       "last_validation_step": self.last_validation_step,
                       "consecutive_skips": self.consecutive_skips,
                       "world_size": self.world_size, "rng_states": states, "best": self.best,
                       "manifest_sha256": self.manifest_hash}
            atomic_save(self.root / filename, payload)
        self.root_call(publish)

    def validate(self, stage):
        preserved_rng = rng_state()
        self.log("validation_started", stage=stage.name, actual_bitstreams=self.config.validation.actual_bitstreams)
        if self.config.validation.actual_bitstreams:
            directory = self.root / "validation" / f"step-{self.global_step:09d}"
            # Release the training parameters, moments and DDP buckets before
            # starting an independent CUDA decoder on the same GPU.
            self.ddp = None
            self.window = None
            for model in self.models.values():
                model.cpu()
            for state in self.optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.cpu()
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            def evaluate():
                image_path = directory / "image.pt"
                video_path = directory / "video.pt" if stage.model == "video" else None
                atomic_save(image_path, model_checkpoint(self.models["image"], self.model_modes.get("image", "float")))
                if video_path is not None:
                    atomic_save(video_path, model_checkpoint(self.models["video"], self.model_modes.get("video", "float")))
                return run_isolated(self.config, stage, image_path, video_path, self.device, directory)
            try:
                result = self.root_call(evaluate)
            finally:
                for model in self.models.values():
                    model.to(self.device)
                for state in self.optimizer.state.values():
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor) and key != "step":
                            state[key] = value.to(self.device)
                self.window = TrainingWindow(self.models[stage.model], stage, self.config,
                                             self.models.get("image") if stage.model == "video" else None).train()
                self.wrap()
        else:
            result = self.root_call(lambda: validate_estimated(
                self.models[stage.model], self.models.get("image") if stage.model == "video" else None,
                self.config, stage, self.records, self.device))
            self.window.train()
        restore_rng(preserved_rng)
        score = result["score"]
        if not math.isfinite(score):
            raise RuntimeError("validation produced non-finite distortion/rate")
        self.log("validation_completed", stage=stage.name, **result)
        self.last_validation_step = self.global_step
        if score < self.best.get(stage.model, float("inf")):
            self.best[stage.model] = score
            self.save(stage, f"best-{stage.model}.pt")
            self.save(stage, "best.pt")
        self.save(stage)

    def run_epoch(self, stage, epoch):
        dataset = TrainingDataset(self.records, stage, self.config.data, self.config.seed)
        dataset.epoch = epoch
        generator = torch.Generator().manual_seed(self.config.seed + epoch)
        if stage.crop_buckets is not None:
            batches = AspectRatioBatchSampler(dataset, self.config.data.batch_per_gpu,
                                               self.world_size, self.rank)
        elif self.distributed:
            sampler = DistributedSampler(dataset, self.world_size, self.rank, seed=self.config.seed, drop_last=False)
            sampler.set_epoch(epoch)
            batches = BatchSampler(sampler, self.config.data.batch_per_gpu, drop_last=True)
        else:
            sampler = RandomSampler(dataset, generator=generator)
            batches = BatchSampler(sampler, self.config.data.batch_per_gpu, drop_last=True)
        total_batches = len(batches)
        if not total_batches:
            raise ValueError("not enough samples for batch_per_gpu; increase stage.samples_per_epoch")
        first_batch = self.position["batch_index"]
        loader = DataLoader(dataset, batch_sampler=RemainingBatches(batches, first_batch),
                            num_workers=self.config.data.num_workers, pin_memory=self.device.type == "cuda",
                            generator=torch.Generator().manual_seed(self.config.seed + epoch + self.rank),
                            persistent_workers=False)
        accumulation = self.config.optimizer.accumulation_steps
        started, log_metrics = time.monotonic(), {}
        for batch_index, frames in enumerate(loader, first_batch):
            frames = frames.to(self.device, non_blocking=True)
            qp = random.randint(0, 63)
            group_start = (batch_index // accumulation) * accumulation
            group_size = min(accumulation, total_batches - group_start)
            group_end = batch_index + 1 == group_start + group_size
            progress = (epoch + group_start / total_batches) / stage.epochs
            lr = stage.min_lr + (stage.lr - stage.min_lr) * (1 + math.cos(math.pi * progress)) / 2
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            chunks = list(temporal_windows(frames, stage))
            reference = None
            for chunk_index, (chunk, start, fraction) in enumerate(chunks):
                synchronize = group_end and chunk_index == len(chunks) - 1
                context = self.ddp.no_sync() if self.distributed and not synchronize else nullcontext()
                with context:
                    loss, metrics, reference = self.ddp(chunk, qp, start, reference)
                    (loss * fraction / group_size).backward()
                for key, value in metrics.items():
                    log_metrics[key] = log_metrics.get(key, 0) + value * fraction / group_size
                del loss, metrics
            del reference, chunks, frames
            if not group_end:
                continue
            norm = torch.nn.utils.clip_grad_norm_(self.models[stage.model].parameters(), self.config.optimizer.gradient_clip)
            nonfinite = self.flag_anywhere(not bool(torch.isfinite(norm)))
            if nonfinite:
                self.consecutive_skips += 1
            else:
                self.optimizer.step()
                self.consecutive_skips = 0
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1
            self.position.update(epoch=epoch, batch_index=batch_index + 1,
                                 stage_step=self.position["stage_step"] + 1)
            if batch_index + 1 == total_batches:
                self.position.update(epoch=epoch + 1, batch_index=0)
            if self.global_step % self.config.output.log_every_steps == 0 or nonfinite:
                numbers = torch.stack([log_metrics[key] for key in sorted(log_metrics)])
                if self.distributed:
                    dist.all_reduce(numbers)
                    numbers /= self.world_size
                values = dict(zip(sorted(log_metrics), numbers.tolist()))
                elapsed = time.monotonic() - started
                op = self.window.primary.ops
                self.log("train_step", stage=stage.name, epoch=epoch, lr=lr, skipped=nonfinite,
                         gradient_norm=float(norm), seconds=elapsed,
                         frames_per_second=group_size * self.config.data.batch_per_gpu * self.world_size
                         * stage.sequence_length / max(elapsed, 1e-9),
                         peak_memory_bytes=torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0,
                         saturated_fraction=float(op.saturated) / max(op.elements, 1) if op.saturated is not None else 0.0,
                         **values)
            log_metrics = {}
            self.window.primary.ops.saturated = None
            self.window.primary.ops.elements = 0
            stopped = self.flag_anywhere(self.stop_requested)
            stage_end = self.position["epoch"] >= stage.epochs or \
                stage.max_steps is not None and self.position["stage_step"] >= stage.max_steps
            if self.global_step % self.config.output.checkpoint_every_steps == 0 or stage_end or stopped \
                    or self.consecutive_skips >= 10:
                self.save(stage)
            if self.consecutive_skips >= 10:
                raise RuntimeError("ten consecutive non-finite optimizer steps; checkpoint saved before stopping")
            if stopped:
                return True
            if self.global_step % self.config.validation.every_steps == 0 or stage_end:
                self.validate(stage)
            started = time.monotonic()
            if stage_end:
                break
        return False

    def run(self):
        previous_handlers = {}
        def stop(signum, frame):
            self.stop_requested = True
        try:
            self.setup()
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[sig] = signal.signal(sig, stop)
            for index in range(self.position["stage_index"], len(self.config.stages)):
                stage = self.config.stages[index]
                self.position["stage_index"] = index
                self.enter_stage(stage)
                self.log("stage_started", stage=stage.name, mode=stage.mode, model=stage.model)
                stage_done = self.position["epoch"] >= stage.epochs or \
                    stage.max_steps is not None and self.position["stage_step"] >= stage.max_steps
                if self.position["stage_step"] and self.last_validation_step != self.global_step \
                        and (stage_done or self.global_step % self.config.validation.every_steps == 0):
                    self.validate(stage)
                for epoch in range(self.position["epoch"], stage.epochs):
                    if stage.max_steps is not None and self.position["stage_step"] >= stage.max_steps:
                        break
                    if self.run_epoch(stage, epoch):
                        self.log("training_interrupted", checkpoint=str(self.root / "last.pt"))
                        return 130
                self.log("stage_completed", stage=stage.name)
                # Persist the completed-stage position before resetting local
                # counters. Resume can safely pass through this stage again.
                self.save(stage)
                self.position.update(stage_index=index + 1, epoch=0, batch_index=0, stage_step=0)
            self.log("training_completed", checkpoint=str(self.root / "last.pt"))
            return 0
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
            if self.lock is not None:
                self.lock.close()
            if self.distributed and dist.is_initialized():
                dist.destroy_process_group()


def train(config, resume=None):
    return Trainer(config, resume).run()
