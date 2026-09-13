import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .extension_loader import load_inference_extensions


FEATURE_SCALE = 512
WEIGHT_SCALE = 8192
INT16_MIN = -32768
INT16_MAX = 32767
INT16_Q_MIN = FEATURE_SCALE // 2


try:
    _INT16_EXT = load_inference_extensions(
        required=("conv2d_int16_cuda",)
    )
    conv2d_int16_cuda = _INT16_EXT.conv2d_int16_cuda
    conv2d_int16_residual_cuda = getattr(_INT16_EXT, "conv2d_int16_residual_cuda", None)
    add_bias_int16_cuda = getattr(_INT16_EXT, "add_bias_int16_cuda", None)
    add_tensors_int16_cuda = getattr(_INT16_EXT, "add_tensors_int16_cuda", None)
    mul_feature_scale_int16_cuda = getattr(_INT16_EXT, "mul_feature_scale_int16_cuda", None)
    reciprocal_scale_int16_cuda = getattr(_INT16_EXT, "reciprocal_scale_int16_cuda", None)
    apply_lut_int16_cuda = getattr(_INT16_EXT, "apply_lut_int16_cuda", None)
    bias_quant_int16_cuda = getattr(_INT16_EXT, "bias_quant_int16_cuda", None)
    wsilu_chunk_add_int16_cuda = getattr(_INT16_EXT, "wsilu_chunk_add_int16_cuda", None)
    bias_wsilu_depthwise_conv2d_int16_cuda = getattr(_INT16_EXT, "bias_wsilu_depthwise_conv2d_int16_cuda", None)
    bias_pixel_shuffle_2_int16_cuda = getattr(_INT16_EXT, "bias_pixel_shuffle_2_int16_cuda", None)
    bias_pixel_shuffle_8_int16_cuda = getattr(_INT16_EXT, "bias_pixel_shuffle_8_int16_cuda", None)
    CUSTOMIZED_INT16_CUDA_INFERENCE = True
except Exception:  # pylint: disable=W0718
    conv2d_int16_cuda = None
    conv2d_int16_residual_cuda = None
    add_bias_int16_cuda = None
    add_tensors_int16_cuda = None
    mul_feature_scale_int16_cuda = None
    reciprocal_scale_int16_cuda = None
    apply_lut_int16_cuda = None
    bias_quant_int16_cuda = None
    wsilu_chunk_add_int16_cuda = None
    bias_wsilu_depthwise_conv2d_int16_cuda = None
    bias_pixel_shuffle_2_int16_cuda = None
    bias_pixel_shuffle_8_int16_cuda = None
    CUSTOMIZED_INT16_CUDA_INFERENCE = False


_BOOL_TRUE = {"1", "true", "yes", "on"}
_AUTOTUNE_RESIDUAL = os.getenv("DCVC_INT16_AUTOTUNE", "0").strip().lower() in _BOOL_TRUE
_RESIDUAL_TILES = {}
_ALLOW_RESIDUAL_AUTOTUNE = ContextVar("allow_residual_autotune", default=True)
_WSILU_LUTS = {}
_PRIOR_LUTS = {}
_SCALE_INDEX_LUTS = {}
_PREPARED_WSILU_LUT_CPU = None
_PREPARED_PRIOR_LUT_CPU = None
_PREPARED_SCALE_INDEX_LUTS_CPU = {}


def int16_inference_enabled():
    return CUSTOMIZED_INT16_CUDA_INFERENCE and \
        os.getenv("DCVC_USE_INT16", "0").strip().lower() in _BOOL_TRUE


def clip_to_int16(x):
    return x.clamp(INT16_MIN, INT16_MAX).to(dtype=torch.int16)


def round_divide(x, denominator):
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.int32)
    x = x.to(dtype=torch.int32)
    if torch.is_tensor(denominator):
        denominator = denominator.to(device=x.device, dtype=torch.int32)
    else:
        # Keep scalar divisors on the host: torch.tensor(..., device="cuda")
        # performs a host-to-device copy that is forbidden during graph capture.
        denominator = int(denominator)
    abs_x = torch.abs(x)
    return torch.where(x >= 0,
                       (abs_x + denominator // 2) // denominator,
                       -((abs_x + denominator // 2) // denominator))


def feature_to_int16(x):
    if x.dtype == torch.int16:
        return x.contiguous()
    return clip_to_int16(torch.round(x.to(dtype=torch.float32) * FEATURE_SCALE).to(dtype=torch.int32))


def feature_to_float(x):
    if x.dtype != torch.int16:
        return x
    return x.to(dtype=torch.float32) / FEATURE_SCALE


def symbol_to_feature(x):
    return clip_to_int16(x.to(dtype=torch.int32) * FEATURE_SCALE)


def bias_to_int16(bias):
    if bias is None:
        return None
    if bias.dtype == torch.int16:
        return bias.contiguous()
    return clip_to_int16(torch.round(bias.to(dtype=torch.float32) * FEATURE_SCALE).to(dtype=torch.int32))


def weight_to_int16(weight):
    if weight.dtype == torch.int16:
        return weight.contiguous()
    return clip_to_int16(torch.round(weight.to(dtype=torch.float32) * WEIGHT_SCALE).to(dtype=torch.int32))


def _cache_key(device):
    return f"{device.type}:{device.index}"


def _scale_index_key(scale_min, scale_max, log_scale_min, log_step_recip):
    return "|".join(
        f"{float(v):.17g}"
        for v in (scale_min, scale_max, log_scale_min, log_step_recip)
    )


def _get_module_cache(module):
    cache = getattr(module, "_int16_cache", None)
    if cache is None:
        cache = {}
        module._int16_cache = cache
    return cache


def prepare_conv_int16_params(conv):
    conv._int16_weight_cpu = weight_to_int16(conv.weight.detach().to(dtype=torch.float32, device="cpu"))
    conv._int16_bias_cpu = None if conv.bias is None else \
        bias_to_int16(conv.bias.detach().to(dtype=torch.float32, device="cpu"))


def prepare_model_int16_convs(module):
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            prepare_conv_int16_params(child)


def prepare_feature_param_int16(module, name, tensor):
    if tensor is None:
        return
    if not hasattr(module, "_prepared_int16_feature_params"):
        module._prepared_int16_feature_params = {}
    module._prepared_int16_feature_params[name] = feature_to_int16(
        tensor.detach().to(dtype=torch.float32, device="cpu")
    ).contiguous()


def get_prepared_feature_param_slice(module, name, tensor, start, end, device):
    prepared = getattr(module, "_prepared_int16_feature_params", None)
    if prepared is None or name not in prepared:
        return feature_to_int16(tensor[start:end]).to(device=device)

    cache_name = "_prepared_int16_feature_param_cache"
    if not hasattr(module, cache_name):
        setattr(module, cache_name, {})
    cache = getattr(module, cache_name)
    key = (name, _cache_key(device))
    if key not in cache:
        cache[key] = prepared[name].to(device=device).contiguous()
    return cache[key][start:end]


def get_quantized_conv_params(conv, device):
    cache = _get_module_cache(conv)
    key = _cache_key(device)
    if key not in cache:
        weight_cpu = getattr(conv, "_int16_weight_cpu", None)
        bias_cpu = getattr(conv, "_int16_bias_cpu", None)
        if weight_cpu is not None:
            weight = weight_cpu.to(device=device).contiguous()
            bias = None if bias_cpu is None else bias_cpu.to(device=device).contiguous()
        else:
            weight = weight_to_int16(conv.weight.detach()).to(device=device).contiguous()
            bias = None if conv.bias is None else \
                bias_to_int16(conv.bias.detach()).to(device=device).contiguous()
        cache[key] = {
            "weight": weight,
            "bias": bias,
        }
    return cache[key]["weight"], cache[key]["bias"]


def _ensure_feature_tensor_int16(x, device):
    if x.dtype != torch.int16:
        x = feature_to_int16(x)
    if x.device != device:
        x = x.to(device=device)
    return x.contiguous()


def _ensure_bias_tensor_int16(bias, device):
    if bias is None:
        return None
    if bias.dtype != torch.int16:
        bias = bias_to_int16(bias)
    if bias.device != device:
        bias = bias.to(device=device)
    return bias.contiguous()


def export_model_int16_state(module):
    convs = {}
    for name, child in module.named_modules():
        if isinstance(child, nn.Conv2d):
            if getattr(child, "_int16_weight_cpu", None) is None:
                prepare_conv_int16_params(child)
            convs[name] = {
                "weight": child._int16_weight_cpu.cpu().contiguous(),
                "bias": None if child._int16_bias_cpu is None
                else child._int16_bias_cpu.cpu().contiguous(),
            }

    prepared_feature_params = {}
    for name, tensor in getattr(module, "_prepared_int16_feature_params", {}).items():
        prepared_feature_params[name] = tensor.cpu().contiguous()

    return {
        "convs": convs,
        "feature_params": prepared_feature_params,
    }


def load_model_int16_state(module, state):
    if state is None:
        return

    named_modules = dict(module.named_modules())
    for name, params in state.get("convs", {}).items():
        conv = named_modules.get(name)
        if not isinstance(conv, nn.Conv2d):
            continue
        conv._int16_weight_cpu = params["weight"].to(dtype=torch.int16, device="cpu").contiguous()
        bias = params.get("bias")
        conv._int16_bias_cpu = None if bias is None else \
            bias.to(dtype=torch.int16, device="cpu").contiguous()
        conv._int16_cache = {}

    feature_params = {}
    for name, tensor in state.get("feature_params", {}).items():
        feature_params[name] = tensor.to(dtype=torch.int16, device="cpu").contiguous()
    if feature_params:
        module._prepared_int16_feature_params = feature_params
        module._prepared_int16_feature_param_cache = {}


def _build_wsilu_lut_cpu():
    values = torch.arange(INT16_MIN, INT16_MAX + 1, dtype=torch.int32)
    values_f = values.to(dtype=torch.float32) / FEATURE_SCALE
    lut = torch.round(values_f * torch.sigmoid(4.0 * values_f) * FEATURE_SCALE).to(dtype=torch.int32)
    return clip_to_int16(lut).cpu().contiguous()


def _build_prior_lut_cpu():
    values = torch.arange(INT16_MIN, INT16_MAX + 1, dtype=torch.int32)
    values_f = values.to(dtype=torch.float32) / FEATURE_SCALE
    lut = torch.round((torch.sigmoid(values_f) * 1.5 + 0.5) * FEATURE_SCALE).to(dtype=torch.int32)
    return clip_to_int16(lut).cpu().contiguous()


def _build_scale_index_lut_cpu(scale_min, scale_max, log_scale_min, log_step_recip):
    values = torch.arange(INT16_MIN, INT16_MAX + 1, dtype=torch.int32)
    values_f = (values.to(dtype=torch.float32) / FEATURE_SCALE).clamp_(scale_min, scale_max)
    lut = torch.floor((torch.log(values_f) - log_scale_min) * log_step_recip)
    return lut.clamp_(0, 255).to(dtype=torch.uint8).cpu().contiguous()


def export_int16_lut_state(scale_index_specs=()):
    scale_index = {}
    for spec in scale_index_specs:
        key = _scale_index_key(*spec)
        if key not in scale_index:
            scale_index[key] = _build_scale_index_lut_cpu(*spec)
    return {
        "wsilu": _build_wsilu_lut_cpu(),
        "prior": _build_prior_lut_cpu(),
        "scale_index": scale_index,
    }


def load_int16_lut_state(state):
    global _PREPARED_WSILU_LUT_CPU, _PREPARED_PRIOR_LUT_CPU, _PREPARED_SCALE_INDEX_LUTS_CPU

    if state is None:
        return

    wsilu = state.get("wsilu")
    prior = state.get("prior")
    scale_index = state.get("scale_index", {})

    _PREPARED_WSILU_LUT_CPU = None if wsilu is None else wsilu.to(dtype=torch.int16, device="cpu").contiguous()
    _PREPARED_PRIOR_LUT_CPU = None if prior is None else prior.to(dtype=torch.int16, device="cpu").contiguous()
    _PREPARED_SCALE_INDEX_LUTS_CPU = {
        key: lut.to(dtype=torch.uint8, device="cpu").contiguous()
        for key, lut in scale_index.items()
    }
    _WSILU_LUTS.clear()
    _PRIOR_LUTS.clear()
    _SCALE_INDEX_LUTS.clear()


def conv2d_module_int16(x, conv):
    if conv2d_int16_cuda is None:
        raise RuntimeError("int16 CUDA convolution is unavailable")
    weight, bias = get_quantized_conv_params(conv, x.device)
    stride_h, stride_w = conv.stride
    pad_h, pad_w = conv.padding
    return conv2d_int16_cuda(x.contiguous(), weight, bias,
                             stride_h, stride_w, pad_h, pad_w, conv.groups)


def conv2d_residual_module_int16(x, conv, residual):
    if conv2d_int16_residual_cuda is not None and conv.kernel_size == (1, 1) and \
            conv.stride == (1, 1) and conv.padding == (0, 0) and conv.groups == 1:
        weight, bias = get_quantized_conv_params(conv, x.device)
        tile = _residual_tile(x, weight, bias, residual) if _AUTOTUNE_RESIDUAL else 32
        return conv2d_int16_residual_cuda(x, weight, bias, residual, tile)
    return add_tensors_int16(conv2d_module_int16(x, conv), residual)


@contextmanager
def residual_autotune_scope(allow_new_shapes):
    """Reuse cached tiles while optionally suppressing measurements of new shapes."""
    token = _ALLOW_RESIDUAL_AUTOTUNE.set(
        _ALLOW_RESIDUAL_AUTOTUNE.get() and allow_new_shapes
    )
    try:
        yield
    finally:
        _ALLOW_RESIDUAL_AUTOTUNE.reset(token)


def _residual_tile(x, weight, bias, residual):
    # Timing choices affect scheduling only. Every tile executes the same exact
    # integer arithmetic, including convolution saturation before residual add.
    key = (x.device, tuple(x.shape), tuple(weight.shape), bias is not None)
    if key in _RESIDUAL_TILES:
        return _RESIDUAL_TILES[key]
    if not _ALLOW_RESIDUAL_AUTOTUNE.get():
        # Do not cache this fallback: a later video may still tune the shape.
        return 32
    with torch.cuda.device(x.device):
        if torch.cuda.is_current_stream_capturing():
            return 32
        timings = {}
        for tile in (16, 32, 64):
            for _ in range(3):
                conv2d_int16_residual_cuda(x, weight, bias, residual, tile)
            samples = []
            for _ in range(3):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(5):
                    conv2d_int16_residual_cuda(x, weight, bias, residual, tile)
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            timings[tile] = sorted(samples)[1]
        selected = min(timings, key=timings.get)
    _RESIDUAL_TILES[key] = selected
    return selected


def conv2d_bias_quant_module_int16(x, conv, quant_step):
    if conv2d_int16_cuda is None:
        raise RuntimeError("int16 CUDA convolution is unavailable")
    weight, bias = get_quantized_conv_params(conv, x.device)
    if bias is None or bias_quant_int16_cuda is None:
        out = conv2d_module_int16(x, conv)
        return mul_feature_scale_int16(out, quant_step)

    quant_step = _ensure_feature_tensor_int16(quant_step, x.device)
    stride_h, stride_w = conv.stride
    pad_h, pad_w = conv.padding
    out = conv2d_int16_cuda(x.contiguous(), weight, None, stride_h, stride_w, pad_h, pad_w, conv.groups)
    return bias_quant_int16_cuda(out.contiguous(), bias, quant_step)


def conv2d_bias_pixel_shuffle_2_module_int16(x, conv):
    if conv2d_int16_cuda is None:
        raise RuntimeError("int16 CUDA convolution is unavailable")
    weight, bias = get_quantized_conv_params(conv, x.device)
    if bias is None or bias_pixel_shuffle_2_int16_cuda is None:
        out = conv2d_module_int16(x, conv)
        return F.pixel_shuffle(out, 2)

    stride_h, stride_w = conv.stride
    pad_h, pad_w = conv.padding
    out = conv2d_int16_cuda(x.contiguous(), weight, None, stride_h, stride_w, pad_h, pad_w, conv.groups)
    return bias_pixel_shuffle_2_int16_cuda(out.contiguous(), bias)


def conv2d_bias_pixel_shuffle_8_module_int16(x, conv):
    if conv2d_int16_cuda is None:
        raise RuntimeError("int16 CUDA convolution is unavailable")
    weight, bias = get_quantized_conv_params(conv, x.device)
    if bias is None or bias_pixel_shuffle_8_int16_cuda is None:
        out = conv2d_module_int16(x, conv)
        out = F.pixel_shuffle(out, 8)
        return torch.clamp(out.to(dtype=torch.int32), 0, FEATURE_SCALE).to(dtype=torch.int16)

    stride_h, stride_w = conv.stride
    pad_h, pad_w = conv.padding
    out = conv2d_int16_cuda(x.contiguous(), weight, None, stride_h, stride_w, pad_h, pad_w, conv.groups)
    return bias_pixel_shuffle_8_int16_cuda(out.contiguous(), bias)


def depthwise_wsilu_module_int16(x, conv):
    if conv2d_int16_cuda is None:
        raise RuntimeError("int16 CUDA convolution is unavailable")
    weight, bias = get_quantized_conv_params(conv, x.device)
    if bias is not None and bias_wsilu_depthwise_conv2d_int16_cuda is not None and \
            conv.groups == x.shape[1] == conv.out_channels and conv.kernel_size == (3, 3) and \
            conv.stride == (1, 1) and conv.padding == (1, 1):
        lut = _get_wsilu_lut(x.device)
        return bias_wsilu_depthwise_conv2d_int16_cuda(
            x.contiguous(), weight, bias, lut
        )
    out = wsilu_int16(x)
    return conv2d_module_int16(out, conv)


def apply_module_int16(module, x):
    if isinstance(module, nn.Conv2d):
        return conv2d_module_int16(x, module)
    if isinstance(module, nn.Sequential):
        for submodule in module:
            x = apply_module_int16(submodule, x)
        return x
    return module(x)


def add_bias_int16(x, bias):
    bias = _ensure_bias_tensor_int16(bias, x.device)
    if bias is None:
        return x
    if add_bias_int16_cuda is not None and x.is_cuda and x.dtype == torch.int16:
        return add_bias_int16_cuda(x.contiguous(), bias)
    while bias.ndim < x.ndim:
        bias = bias.unsqueeze(-1)
    return clip_to_int16(x.to(dtype=torch.int32) + bias.to(dtype=torch.int32))


def add_tensors_int16(x, y):
    if add_tensors_int16_cuda is not None and x.is_cuda and x.dtype == torch.int16 and y.is_cuda and y.dtype == torch.int16:
        return add_tensors_int16_cuda(x.contiguous(), y.contiguous())
    return clip_to_int16(x.to(dtype=torch.int32) + y.to(dtype=torch.int32))


def mul_feature_scale_int16(x, scale):
    scale = _ensure_feature_tensor_int16(scale, x.device)
    if mul_feature_scale_int16_cuda is not None and x.is_cuda and x.dtype == torch.int16:
        return mul_feature_scale_int16_cuda(x.contiguous(), scale.contiguous())
    out = round_divide(x.to(dtype=torch.int32) * scale.to(dtype=torch.int32), FEATURE_SCALE)
    return clip_to_int16(out)


def clamp_quant_step_int16(q):
    return torch.clamp(q.to(dtype=torch.int32), min=INT16_Q_MIN).to(dtype=torch.int16)


def _apply_lut(x, lut):
    index = (x.to(dtype=torch.int32) + 32768).to(dtype=torch.int64)
    return lut[index]


def _get_wsilu_lut(device):
    key = _cache_key(device)
    if key not in _WSILU_LUTS:
        lut_cpu = _PREPARED_WSILU_LUT_CPU if _PREPARED_WSILU_LUT_CPU is not None else _build_wsilu_lut_cpu()
        _WSILU_LUTS[key] = lut_cpu.to(device=device)
    return _WSILU_LUTS[key]


def wsilu_int16(x):
    lut = _get_wsilu_lut(x.device)
    if apply_lut_int16_cuda is not None and x.is_cuda and x.dtype == torch.int16:
        return apply_lut_int16_cuda(x.contiguous(), lut)
    return _apply_lut(x, lut)


def wsilu_chunk_add_int16(x):
    lut = _get_wsilu_lut(x.device)
    if wsilu_chunk_add_int16_cuda is not None and x.is_cuda and x.dtype == torch.int16:
        return wsilu_chunk_add_int16_cuda(x.contiguous(), lut)
    x1, x2 = wsilu_int16(x).chunk(2, 1)
    return clip_to_int16(x1.to(dtype=torch.int32) + x2.to(dtype=torch.int32))


def _get_prior_lut(device):
    key = _cache_key(device)
    if key not in _PRIOR_LUTS:
        lut_cpu = _PREPARED_PRIOR_LUT_CPU if _PREPARED_PRIOR_LUT_CPU is not None else _build_prior_lut_cpu()
        _PRIOR_LUTS[key] = lut_cpu.to(device=device)
    return _PRIOR_LUTS[key]


def prior_quant_step_int16(x):
    lut = _get_prior_lut(x.device)
    if apply_lut_int16_cuda is not None and x.is_cuda and x.dtype == torch.int16:
        return apply_lut_int16_cuda(x.contiguous(), lut)
    return _apply_lut(x, lut)


def get_scale_index_lut(device, scale_min, scale_max, log_scale_min, log_step_recip):
    spec_key = _scale_index_key(scale_min, scale_max, log_scale_min, log_step_recip)
    cache_key = (_cache_key(device), spec_key)
    if cache_key not in _SCALE_INDEX_LUTS:
        lut_cpu = _PREPARED_SCALE_INDEX_LUTS_CPU.get(spec_key)
        if lut_cpu is None:
            lut_cpu = _build_scale_index_lut_cpu(scale_min, scale_max, log_scale_min, log_step_recip)
        _SCALE_INDEX_LUTS[cache_key] = lut_cpu.to(device=device)
    return _SCALE_INDEX_LUTS[cache_key]


def scale_to_index_int16(scales, scale_min, scale_max, log_scale_min, log_step_recip):
    lut = get_scale_index_lut(scales.device, scale_min, scale_max, log_scale_min, log_step_recip)
    index = (scales.to(dtype=torch.int32) + 32768).to(dtype=torch.int64)
    return lut[index]


def round_feature_to_symbols(x):
    rounded = round_divide(x.to(dtype=torch.int32), FEATURE_SCALE).clamp_(-128, 127)
    return rounded.to(dtype=torch.int16), rounded.to(dtype=torch.int8)


def reciprocal_scale_int16(q_dec, min_val):
    if reciprocal_scale_int16_cuda is not None and q_dec.is_cuda and q_dec.dtype == torch.int16:
        return reciprocal_scale_int16_cuda(q_dec.contiguous(), min_val)
    min_q = int(round(min_val * FEATURE_SCALE))
    q_dec = torch.clamp(q_dec.to(dtype=torch.int32), min=min_q)
    reciprocal = round_divide(FEATURE_SCALE * FEATURE_SCALE, q_dec)
    return q_dec.to(dtype=torch.int16), clip_to_int16(reciprocal)


def optional_int16(mode_tensor: Optional[torch.Tensor], device):
    if mode_tensor is None:
        return None
    return feature_to_int16(mode_tensor).to(device=device)
