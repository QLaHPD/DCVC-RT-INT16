from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np
import torch

from src.models.image_model import DMCI
from src.models.video_model import DMC
from src.layers import int16_inference as int16_runtime
from src.utils.common import load_model_for_inference, set_torch_env
from src.utils.stream_helper import (
    BitstreamInfo,
    NalType,
    SPSHelper,
    read_header,
    read_ip_remaining,
    read_sps_remaining,
    skip_ip_remaining,
)
from src.utils.transforms import ycbcr2rgb, yuv_444_to_420


OutputFormat = Optional[str]


@dataclass(frozen=True)
class FrameRecord:
    index: int
    offset: int
    nal_type: NalType
    sps_id: int
    sps: Dict[str, int]
    qp: int
    bitstream_length: int
    sync_index: int

    @property
    def frame_type(self) -> str:
        return "I" if self.nal_type == NalType.NAL_I else "P"


@dataclass(frozen=True)
class BitstreamFrameIndex:
    path: Path
    info: BitstreamInfo
    frames: List[FrameRecord]

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def width(self) -> int:
        return self.info.width

    @property
    def height(self) -> int:
        return self.info.height

    def validate_frame_index(self, frame_index: int):
        if frame_index < 0 or frame_index >= self.frame_count:
            raise IndexError(f"frame index {frame_index} outside 0..{self.frame_count - 1}")

    def dependency_start(self, frame_index: int) -> int:
        self.validate_frame_index(frame_index)
        return self.frames[frame_index].sync_index


@dataclass
class DecoderModels:
    i_frame_net: DMCI
    p_frame_net: Optional[DMC]
    device: torch.device


@dataclass
class DecodedFrame:
    index: int
    frame_type: str
    width: int
    height: int
    qp: int
    sps: Dict[str, int]
    y: Optional[np.ndarray] = None
    uv: Optional[np.ndarray] = None
    rgb: Optional[np.ndarray] = None
    tensor: Optional[torch.Tensor] = None


def resolve_decode_device(use_cuda: bool = True, cuda_idx: Optional[int] = None) -> torch.device:
    if use_cuda and torch.cuda.is_available():
        if cuda_idx is None:
            cuda_idx = 0
        torch.cuda.set_device(cuda_idx)
        return torch.device(f"cuda:{cuda_idx}")
    return torch.device("cpu")


def configure_decode_runtime(requested: str, use_cuda: bool) -> str:
    """Select and verify decoder arithmetic before loading model weights."""

    requested = str(requested or "auto").lower()
    if requested not in {"auto", "int16", "float"}:
        raise ValueError(f"unsupported decoder runtime: {requested}")
    if requested == "int16":
        if not use_cuda:
            raise ValueError("INT16 decoding requires --cuda true")
        if not torch.cuda.is_available():
            raise ValueError("INT16 decoding requires an available CUDA device")
        if not int16_runtime.CUSTOMIZED_INT16_CUDA_INFERENCE:
            raise ValueError(
                "INT16 decoding was requested, but the INT16 CUDA extension is unavailable; "
                "build it on this machine with ./build_native_extensions.sh"
            )
        os.environ["DCVC_USE_INT16"] = "1"
        if not int16_runtime.int16_inference_enabled():
            raise ValueError("INT16 decoding was requested but could not be enabled")
        return "CUDA INT16"
    if requested == "float":
        os.environ["DCVC_USE_INT16"] = "0"
        return "CUDA float" if use_cuda and torch.cuda.is_available() else "CPU float"

    if int16_runtime.int16_inference_enabled() and use_cuda and torch.cuda.is_available():
        return "CUDA INT16"
    return "CUDA float" if use_cuda and torch.cuda.is_available() else "CPU float"


def load_decoder_models(
    model_path_i: str,
    model_path_p: str,
    device: torch.device | str,
    force_zero_thres: Optional[float] = None,
    load_p_model: bool = True,
) -> DecoderModels:
    set_torch_env()
    device = torch.device(device)
    i_frame_net = load_model_for_inference(DMCI, model_path_i, device, force_zero_thres)
    p_frame_net = (
        load_model_for_inference(DMC, model_path_p, device, force_zero_thres)
        if load_p_model else None
    )
    return DecoderModels(i_frame_net=i_frame_net, p_frame_net=p_frame_net, device=device)


def build_bitstream_index(path: str | Path) -> BitstreamFrameIndex:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Invalid bitstream file "{path}".')

    frames: List[FrameRecord] = []
    sps_helper = SPSHelper()
    width = 0
    height = 0
    current_sync = 0

    with path.open("rb") as f:
        while True:
            offset = f.tell()
            try:
                header = read_header(f)
            except struct.error:
                break

            nal_type = header["nal_type"]
            if nal_type == NalType.NAL_SPS:
                sps = read_sps_remaining(f, header["sps_id"])
                sps_helper.add_sps_by_id(sps)
                if width <= 0 or height <= 0:
                    width = sps["width"]
                    height = sps["height"]
                continue

            if nal_type not in (NalType.NAL_I, NalType.NAL_P):
                break

            sps = sps_helper.get_sps_by_id(header["sps_id"])
            if sps is None:
                raise ValueError(f"frame at byte {offset} references missing SPS id {header['sps_id']}")
            if width <= 0 or height <= 0:
                width = sps["width"]
                height = sps["height"]

            qp, bitstream_length = skip_ip_remaining(f)
            frame_index = len(frames)
            if nal_type == NalType.NAL_I:
                current_sync = frame_index

            frames.append(FrameRecord(
                index=frame_index,
                offset=offset,
                nal_type=nal_type,
                sps_id=header["sps_id"],
                sps=sps.copy(),
                qp=qp,
                bitstream_length=bitstream_length,
                sync_index=current_sync,
            ))

    return BitstreamFrameIndex(
        path=path,
        info=BitstreamInfo(frame_count=len(frames), width=width, height=height),
        frames=frames,
    )


def tensor_to_yuv420_uint8(x_hat: torch.Tensor, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    x_hat = x_hat[:, :, :height, :width]
    y_rec, uv_rec = yuv_444_to_420(x_hat)
    y_rec_np = torch.clamp(y_rec * 255, 0, 255).byte().squeeze().cpu().numpy()
    uv_rec_np = torch.clamp(uv_rec * 255, 0, 255).byte().squeeze().cpu().numpy()
    return y_rec_np, uv_rec_np


def tensor_to_rgb_uint8(x_hat: torch.Tensor, width: int, height: int) -> np.ndarray:
    x_hat = x_hat[:, :, :height, :width]
    rgb = ycbcr2rgb(x_hat)
    return torch.clamp(rgb * 255, 0, 255).byte().squeeze(0).permute(1, 2, 0).cpu().numpy()


class BitstreamFrameSource:
    """Stateful decoder for frame-accurate access to one DCVC-RT bitstream.

    P-frames depend on prior decoded state, so seeking positions the decoder at
    the nearest previous I-frame and decodes dependencies up to the target.
    """

    def __init__(
        self,
        bin_path: str | Path,
        models: DecoderModels,
        frame_index: Optional[BitstreamFrameIndex] = None,
    ):
        self.index = frame_index or build_bitstream_index(bin_path)
        self.models = models
        self._file = self.index.path.open("rb")
        self._next_index: Optional[int] = None

    def __enter__(self) -> "BitstreamFrameSource":
        return self

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        self.close()

    def close(self):
        if not self._file.closed:
            self._file.close()

    @property
    def next_index(self) -> Optional[int]:
        return self._next_index

    def reset(self, frame_index: int = 0):
        if self.index.frame_count == 0:
            self._next_index = 0
            return
        self.index.validate_frame_index(frame_index)
        if self.models.p_frame_net is not None:
            self.models.p_frame_net.clear_dpb()
            self.models.p_frame_net.set_curr_poc(frame_index)
        elif any(record.nal_type == NalType.NAL_P for record in self.index.frames):
            raise ValueError("predictive bitstream requires the video model")
        self._next_index = frame_index

    def seek(self, frame_index: int, output_format: str = "rgb") -> DecodedFrame:
        self._position_for_frame(frame_index)
        frame = self._decode_next(output_format)
        if frame is None:
            raise RuntimeError("internal decoder error: selected frame was discarded")
        return frame

    def read_next(self, output_format: str = "rgb") -> Optional[DecodedFrame]:
        if self._next_index is None:
            if self.index.frame_count == 0:
                return None
            self.reset(0)
        if self._next_index is None or self._next_index >= self.index.frame_count:
            return None
        return self._decode_next(output_format)

    def decode_frames(self, frame_indices: Sequence[int], output_format: str = "rgb") -> List[DecodedFrame]:
        requested = list(frame_indices)
        if not requested:
            return []

        decoded = {
            frame.index: frame
            for frame in self.iter_frames(frame_indices=requested, output_format=output_format)
        }
        return [decoded[index] for index in requested]

    def iter_frames(
        self,
        start_frame: int = 0,
        stop_frame: Optional[int] = None,
        frame_indices: Optional[Iterable[int]] = None,
        output_format: str = "rgb",
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Iterator[DecodedFrame]:
        if self.index.frame_count == 0:
            return

        targets = None
        if frame_indices is not None:
            targets = set(frame_indices)
            if not targets:
                return
            for frame_index in targets:
                self.index.validate_frame_index(frame_index)
            start_frame = min(targets)
            stop_frame = max(targets) + 1

        self.index.validate_frame_index(start_frame)
        if stop_frame is None:
            stop_frame = self.index.frame_count
        stop_frame = min(stop_frame, self.index.frame_count)

        self._position_for_frame(start_frame)
        while self._next_index is not None and self._next_index < stop_frame:
            if should_stop is not None and should_stop():
                break
            current = self._next_index
            materialize = targets is None or current in targets
            frame = self._decode_next(output_format if materialize else None)
            if frame is not None:
                yield frame

    def _position_for_frame(self, frame_index: int):
        self.index.validate_frame_index(frame_index)
        sync_index = self.index.dependency_start(frame_index)

        should_reset = self._next_index is None or frame_index < self._next_index
        if not should_reset and self._next_index is not None:
            from_current = frame_index - self._next_index
            from_sync = frame_index - sync_index
            should_reset = from_sync < from_current

        if should_reset:
            self.reset(sync_index)

        while self._next_index is not None and self._next_index < frame_index:
            self._decode_next(None)

    def _decode_next(self, output_format: OutputFormat) -> Optional[DecodedFrame]:
        if self._next_index is None:
            self.reset(0)
        if self._next_index is None or self._next_index >= self.index.frame_count:
            return None

        record = self.index.frames[self._next_index]
        decoded = self._decode_record(record)
        self._next_index += 1

        if output_format is None or output_format == "none":
            return None
        return self._materialize_frame(record, decoded["x_hat"], output_format)

    @torch.inference_mode()
    def _decode_record(self, record: FrameRecord) -> Dict[str, torch.Tensor]:
        self._file.seek(record.offset)
        header = read_header(self._file)
        if header["nal_type"] != record.nal_type:
            raise ValueError(f"bitstream changed while decoding frame {record.index}")
        qp, bit_stream = read_ip_remaining(self._file)
        if qp != record.qp:
            raise ValueError(f"bitstream QP changed while decoding frame {record.index}")

        if record.nal_type == NalType.NAL_I:
            decoded = self.models.i_frame_net.decompress(bit_stream, record.sps, qp)
            if self.models.p_frame_net is not None:
                self.models.p_frame_net.clear_dpb()
                self.models.p_frame_net.add_ref_frame(None, decoded["x_hat"])
            return decoded

        if self.models.p_frame_net is None:
            raise ValueError("predictive bitstream requires the video model")
        if record.sps["use_ada_i"]:
            self.models.p_frame_net.reset_ref_feature()
        return self.models.p_frame_net.decompress(bit_stream, record.sps, qp)

    def _materialize_frame(
        self,
        record: FrameRecord,
        x_hat: torch.Tensor,
        output_format: str,
    ) -> DecodedFrame:
        width = record.sps["width"]
        height = record.sps["height"]
        output_format = output_format.lower()

        y = None
        uv = None
        rgb = None
        tensor = None

        if output_format in {"yuv420", "both"}:
            y, uv = tensor_to_yuv420_uint8(x_hat, width, height)
        if output_format in {"rgb", "both"}:
            rgb = tensor_to_rgb_uint8(x_hat, width, height)
        if output_format == "tensor":
            tensor = x_hat[:, :, :height, :width].detach()
        if output_format not in {"yuv420", "rgb", "both", "tensor"}:
            raise ValueError(f"unsupported output format: {output_format}")

        return DecodedFrame(
            index=record.index,
            frame_type=record.frame_type,
            width=width,
            height=height,
            qp=record.qp,
            sps=record.sps.copy(),
            y=y,
            uv=uv,
            rgb=rgb,
            tensor=tensor,
        )
