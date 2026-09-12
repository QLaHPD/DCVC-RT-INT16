"""Strict, versioned configuration without executable Python config files."""

from dataclasses import asdict, dataclass, field, fields
import json
import math
from pathlib import Path
from typing import Optional

from src.models.architecture import ImageArchitecture, VideoArchitecture


def mapping(cls, value, location):
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a mapping")
    unknown = set(value) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"unknown {location} fields: {', '.join(sorted(unknown))}")
    try:
        return cls(**value)
    except TypeError as exc:
        raise ValueError(f"invalid {location}: {exc}") from exc


def positive(value, name, *, integer=False, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if integer and type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be {'nonnegative' if zero else 'positive'} and finite")


@dataclass
class SourceConfig:
    type: str
    path: str
    split: str = "auto"

    def __post_init__(self):
        if self.type not in {"images", "frame_sequences", "videos"}:
            raise ValueError(f"unsupported data source type: {self.type}")
        if self.split not in {"auto", "train", "validation"}:
            raise ValueError(f"unsupported split: {self.split}")
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("data source path is required")


@dataclass
class DataConfig:
    manifest: str = "data/training/manifest.jsonl"
    cache_root: str = "data/training/cache"
    sources: list = field(default_factory=list)
    validation_fraction: float = 0.1
    num_workers: int = 2
    batch_per_gpu: int = 1
    resize_short_edge: Optional[int] = None
    horizontal_flip: bool = True
    video_loading: str = "cache"

    def __post_init__(self):
        if self.video_loading not in {"cache", "direct"}:
            raise ValueError("data.video_loading must be cache or direct")
        self.sources = [s if isinstance(s, SourceConfig) else mapping(SourceConfig, s, "data.sources")
                        for s in self.sources]
        if not 0 < self.validation_fraction < 1:
            raise ValueError("data.validation_fraction must be between 0 and 1")
        positive(self.num_workers, "data.num_workers", integer=True, zero=True)
        positive(self.batch_per_gpu, "data.batch_per_gpu", integer=True)
        if self.resize_short_edge is not None:
            positive(self.resize_short_edge, "data.resize_short_edge", integer=True)
        if type(self.horizontal_flip) is not bool:
            raise ValueError("data.horizontal_flip must be boolean")


@dataclass
class ModelConfig:
    image: dict = field(default_factory=dict)
    video: dict = field(default_factory=dict)
    init_image: Optional[str] = None
    init_video: Optional[str] = None

    def __post_init__(self):
        self.image = ImageArchitecture.from_dict(self.image).to_dict()
        self.video = VideoArchitecture.from_dict(self.video).to_dict()


@dataclass
class StageConfig:
    name: str
    model: str
    mode: str
    epochs: int
    lr: float
    min_lr: float = 0.0
    crop_size: list = field(default_factory=lambda: [256, 256])
    sequence_length: int = 1
    temporal_gradient_length: int = 1
    samples_per_epoch: Optional[int] = None
    max_steps: Optional[int] = None
    crop_buckets: Optional[list] = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("stage.name is required")
        if self.model not in {"image", "video"} or self.mode not in {"float", "int16"}:
            raise ValueError("stage model must be image/video and mode must be float/int16")
        positive(self.epochs, "stage.epochs", integer=True)
        positive(self.lr, "stage.lr")
        positive(self.min_lr, "stage.min_lr", zero=True)
        if self.min_lr > self.lr:
            raise ValueError("stage.min_lr must not exceed lr")
        if not isinstance(self.crop_size, (list, tuple)) or len(self.crop_size) != 2:
            raise ValueError("stage.crop_size must be [height, width]")
        for size in self.crop_size:
            positive(size, "stage.crop_size", integer=True)
            if size % 16:
                raise ValueError("training crop dimensions must be divisible by 16")
        if self.crop_buckets is not None:
            if self.model != "image" or not isinstance(self.crop_buckets, list) or not self.crop_buckets:
                raise ValueError("stage.crop_buckets requires a nonempty list for an image stage")
            for shape in self.crop_buckets:
                if not isinstance(shape, (list, tuple)) or len(shape) != 2:
                    raise ValueError("each crop bucket must be [height, width]")
                for size in shape:
                    positive(size, "stage.crop_buckets", integer=True)
                    if size % 16:
                        raise ValueError("crop bucket dimensions must be divisible by 16")
            if len({tuple(shape) for shape in self.crop_buckets}) != len(self.crop_buckets):
                raise ValueError("crop buckets must be unique")
        positive(self.sequence_length, "stage.sequence_length", integer=True)
        positive(self.temporal_gradient_length, "stage.temporal_gradient_length", integer=True)
        if self.model == "image" and self.sequence_length != 1:
            raise ValueError("image stages require sequence_length: 1")
        if self.model == "video" and self.sequence_length < 2:
            raise ValueError("video stages require at least two frames")
        for name in ("samples_per_epoch", "max_steps"):
            if getattr(self, name) is not None:
                positive(getattr(self, name), f"stage.{name}", integer=True)


@dataclass
class OptimizerConfig:
    weight_decay: float = 0.0
    betas: list = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    gradient_clip: float = 0.1
    accumulation_steps: int = 1
    activation_checkpointing: bool = False

    def __post_init__(self):
        positive(self.weight_decay, "optimizer.weight_decay", zero=True)
        positive(self.eps, "optimizer.eps")
        positive(self.gradient_clip, "optimizer.gradient_clip")
        positive(self.accumulation_steps, "optimizer.accumulation_steps", integer=True)
        if len(self.betas) != 2 or any(not 0 <= b < 1 for b in self.betas):
            raise ValueError("optimizer.betas must contain two values in [0, 1)")
        if type(self.activation_checkpointing) is not bool:
            raise ValueError("optimizer.activation_checkpointing must be boolean")


@dataclass
class QuantizationConfig:
    feature_scale: int = 512
    weight_scale: int = 8192
    force_zero_thres: Optional[float] = None

    def __post_init__(self):
        if self.feature_scale != 512 or self.weight_scale != 8192:
            raise ValueError("this runtime requires feature_scale=512 and weight_scale=8192")
        if self.force_zero_thres is not None:
            positive(self.force_zero_thres, "quantization.force_zero_thres", zero=True)


@dataclass
class LossConfig:
    image_lambdas: list = field(default_factory=lambda: [10.0, 2048.0])
    video_lambdas: list = field(default_factory=lambda: [1.0, 768.0])
    yuv_weight: float = 0.8
    yuv_channel_weights: list = field(default_factory=lambda: [6.0, 1.0, 1.0])
    frame_weights: list = field(default_factory=lambda: [1.0] * 8)

    def __post_init__(self):
        for name in ("image_lambdas", "video_lambdas"):
            values = getattr(self, name)
            if len(values) != 2:
                raise ValueError(f"loss.{name} requires minimum and maximum")
            for value in values:
                positive(value, f"loss.{name}")
            if values[0] > values[1]:
                raise ValueError(f"loss.{name} must be increasing")
        if not 0 <= self.yuv_weight <= 1:
            raise ValueError("loss.yuv_weight must be in [0, 1]")
        for name, count in (("yuv_channel_weights", 3), ("frame_weights", 8)):
            values = getattr(self, name)
            if len(values) != count:
                raise ValueError(f"loss.{name} requires {count} values")
            for value in values:
                positive(value, f"loss.{name}")


@dataclass
class ValidationConfig:
    qps: list = field(default_factory=lambda: [0, 21, 42, 63])
    every_steps: int = 500
    max_samples: int = 8
    max_frames: int = 65
    actual_bitstreams: bool = True
    reset_interval: int = 32
    image_mode: str = "crop"
    image_max_side: Optional[int] = None

    def __post_init__(self):
        if self.image_mode not in {"crop", "full"}:
            raise ValueError("validation.image_mode must be crop or full")
        if self.image_max_side is not None:
            positive(self.image_max_side, "validation.image_max_side", integer=True)
            if self.image_mode != "full":
                raise ValueError("validation.image_max_side requires image_mode: full")
        if not self.qps or any(type(q) is not int or not 0 <= q <= 63 for q in self.qps):
            raise ValueError("validation.qps must contain integers in [0, 63]")
        for name in ("every_steps", "max_samples", "max_frames"):
            positive(getattr(self, name), f"validation.{name}", integer=True)
        if type(self.reset_interval) is not int or self.reset_interval not in {-1, 0} and self.reset_interval < 1:
            raise ValueError("validation.reset_interval must be -1, 0 or positive")
        if type(self.actual_bitstreams) is not bool:
            raise ValueError("validation.actual_bitstreams must be boolean")


@dataclass
class OutputConfig:
    root: str = "runs/training"
    checkpoint_every_steps: int = 100
    log_every_steps: int = 10

    def __post_init__(self):
        positive(self.checkpoint_every_steps, "output.checkpoint_every_steps", integer=True)
        positive(self.log_every_steps, "output.log_every_steps", integer=True)


@dataclass
class TrainingConfig:
    version: int = 1
    task: str = "pair"
    seed: int = 0
    device: str = "cuda"
    distributed_timeout_seconds: int = 21600
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    stages: list = field(default_factory=list)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def __post_init__(self):
        if type(self.version) is not int or self.version != 1:
            raise ValueError("unsupported training config version")
        if self.task not in {"image", "video", "pair"}:
            raise ValueError("task must be image, video, or pair")
        if self.device not in {"cuda", "cpu"}:
            raise ValueError("device must be cuda or cpu (float smoke tests only)")
        positive(self.seed, "seed", integer=True, zero=True)
        positive(self.distributed_timeout_seconds, "distributed_timeout_seconds", integer=True)
        for name, cls in (("model", ModelConfig), ("data", DataConfig), ("optimizer", OptimizerConfig),
                          ("quantization", QuantizationConfig), ("loss", LossConfig),
                          ("validation", ValidationConfig), ("output", OutputConfig)):
            if not isinstance(getattr(self, name), cls):
                setattr(self, name, mapping(cls, getattr(self, name), name))
        self.stages = [s if isinstance(s, StageConfig) else mapping(StageConfig, s, "stage")
                       for s in self.stages]
        if not self.stages:
            raise ValueError("at least one explicit training stage is required")
        if len({s.name for s in self.stages}) != len(self.stages):
            raise ValueError("stage names must be unique")
        kinds = [s.model for s in self.stages]
        if self.task != "pair" and any(kind != self.task for kind in kinds):
            raise ValueError("stage model does not match task")
        if self.task == "pair":
            if set(kinds) != {"image", "video"} or kinds != sorted(kinds):
                raise ValueError("pair requires image stages followed by video stages")
        if self.task == "video" and not self.model.init_image:
            raise ValueError("video training requires model.init_image")
        if self.device == "cpu" and (any(s.mode == "int16" for s in self.stages)
                                      or self.validation.actual_bitstreams):
            raise ValueError("INT16 training and real-bitstream validation require CUDA")
        if "video" in kinds and self.validation.actual_bitstreams and self.validation.max_frames < 2:
            raise ValueError("video bitstream validation requires validation.max_frames >= 2")

    def to_dict(self):
        return asdict(self)


def load_config(path):
    path = Path(path).resolve()
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        raw = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML training configs require: pip install -r requirements-training.txt") from exc
        raw = yaml.safe_load(text)
    config = mapping(TrainingConfig, raw, "config")
    # Resolve relative paths against the configuration, not the launching shell.
    def resolve(value):
        if value is None:
            return None
        p = Path(value).expanduser()
        return str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
    for key in ("manifest", "cache_root"):
        setattr(config.data, key, resolve(getattr(config.data, key)))
    for source in config.data.sources:
        source.path = resolve(source.path)
    for key in ("init_image", "init_video"):
        setattr(config.model, key, resolve(getattr(config.model, key)))
    config.output.root = resolve(config.output.root)
    return config
