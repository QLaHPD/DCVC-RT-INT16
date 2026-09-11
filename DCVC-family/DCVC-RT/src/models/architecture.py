"""Serializable DCVC-RT topology parameters (also used by inference)."""

from dataclasses import asdict, dataclass, fields
from typing import Mapping


class _Architecture:
    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or value < 1:
                raise ValueError(f"model.{field.name} must be a positive integer")
        divisor = 4 if isinstance(self, ImageArchitecture) else 2
        if self.latent_channels % divisor:
            raise ValueError(f"latent_channels must be divisible by {divisor}")

    @classmethod
    def from_dict(cls, value=None):
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ValueError(f"{cls.__name__} must be a mapping")
        unknown = set(value) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown model fields: {', '.join(sorted(unknown))}")
        return cls(**value)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ImageArchitecture(_Architecture):
    channels: int = 368
    latent_channels: int = 256
    hyper_channels: int = 128
    encoder_blocks: int = 6
    decoder_blocks: int = 12
    fusion_blocks: int = 3
    spatial_blocks: int = 3


@dataclass(frozen=True)
class VideoArchitecture(_Architecture):
    channels: int = 256
    recon_channels: int = 320
    latent_channels: int = 128
    hyper_channels: int = 128
    feature_blocks_1: int = 2
    feature_blocks_2: int = 4
    encoder_blocks: int = 2
    decoder_blocks: int = 3
    recon_blocks: int = 4
    fusion_blocks: int = 3
    spatial_blocks: int = 2


def architecture_metadata(model):
    config = model.architecture
    return {"version": 1, "kind": "image" if isinstance(config, ImageArchitecture) else "video",
            "params": config.to_dict()}


def create_model(kind, architecture=None):
    """Construct without inference preparation or allocating CUDA state."""
    if kind == "image":
        from .image_model import DMCI
        return DMCI(config=ImageArchitecture.from_dict(architecture))
    if kind == "video":
        from .video_model import DMC
        return DMC(config=VideoArchitecture.from_dict(architecture))
    raise ValueError(f"unknown model kind: {kind}")


def model_from_metadata(metadata, expected_kind):
    if metadata is None:
        return create_model(expected_kind)
    if not isinstance(metadata, Mapping) or metadata.get("version") != 1:
        raise ValueError("unsupported model architecture metadata")
    if set(metadata) != {"version", "kind", "params"} or not isinstance(metadata["params"], Mapping):
        raise ValueError("architecture metadata requires version, kind and a params mapping")
    if metadata.get("kind") != expected_kind:
        raise ValueError(f"expected an {expected_kind} checkpoint, got {metadata.get('kind')}")
    return create_model(expected_kind, metadata.get("params"))
