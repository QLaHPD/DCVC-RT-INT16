"""UF integer arithmetic v1. No floating-point operations on runtime features."""

import torch
import torch.nn.functional as F

FEATURE_SCALE = 512
WEIGHT_SCALE = 8192
ARITHMETIC_ID = 'uf-int16-q9-w13-acc64-rna-sat-v1'


def saturate(x):
    return x.clamp(-32768, 32767).to(torch.int16)


def divide_round(x, denominator):
    """Nearest integer, exact ties away from zero; intermediates are int64."""
    x = x.to(torch.int64)
    magnitude = torch.div(x.abs() + denominator // 2, denominator, rounding_mode='floor')
    return torch.where(x < 0, -magnitude, magnitude)


def quantize_parameter(x, scale=FEATURE_SCALE):
    """Preparation only: CPU checkpoint values, nearest-even rounding."""
    values = x.detach().cpu().double()
    if not torch.isfinite(values).all():
        raise ValueError('Cannot prepare non-finite checkpoint parameters')
    return saturate(torch.round(values * scale).to(torch.int64))


def add(*values):
    if values[0].is_cuda and 2 <= len(values) <= 4:
        from uf_int16_cuda import add as native_add
        return native_add(list(values))
    result = values[0].to(torch.int64)
    for value in values[1:]:
        result = result + value.to(torch.int64)
    return saturate(result)


def multiply(x, scale):
    if x.is_cuda:
        from uf_int16_cuda import multiply as native_multiply
        return native_multiply(x, scale)
    return saturate(divide_round(x.to(torch.int64) * scale.to(torch.int64), FEATURE_SCALE))


def reciprocal(x):
    return saturate(divide_round(torch.full_like(x, FEATURE_SCALE**2, dtype=torch.int64),
                                x.to(torch.int64).clamp_min(FEATURE_SCALE // 2)))


def symbols(x):
    return divide_round(x, FEATURE_SCALE).clamp(-128, 127).to(torch.int8)


def from_symbols(x):
    return saturate(x.to(torch.int64) * FEATURE_SCALE)


def from_bytes(x):
    # Source boundary represents byte/255 - 0.5 without floating point.
    if x.is_cuda:
        from uf_int16_cuda import from_bytes as native_bytes
        return native_bytes(x)
    return saturate(divide_round((2 * x.to(torch.int64) - 255) * FEATURE_SCALE, 510))


def lookup(x, lut):
    if x.is_cuda:
        from uf_int16_cuda import lookup as native_lookup
        return native_lookup(x, lut)
    return lut[x.long()+32768]


def wsilu4(x, lut):
    if x.is_cuda:
        from uf_int16_cuda import wsilu4 as native_wsilu4
        return native_wsilu4(x, lut)
    out = lookup(x, lut)
    return add(*(out[:, index::4] for index in range(4)))


def conv2d_reference(x, weight, bias=None, stride=(1, 1), padding=(0, 0), groups=1):
    """Independent CPU int64 convolution; also defines CUDA's numerical contract."""
    if x.device.type != 'cpu' or weight.device.type != 'cpu':
        raise ValueError('Reference convolution requires CPU tensors')
    if x.dtype != torch.int16 or weight.dtype != torch.int16:
        raise TypeError('Convolution features and weights must be int16')
    kh, kw = weight.shape[2:]
    padded = F.pad(x, (padding[1], padding[1], padding[0], padding[0]))
    patches = padded.unfold(2, kh, stride[0]).unfold(3, kw, stride[1])
    b, c, h, w = patches.shape[:4]
    co = weight.shape[0]
    patches = patches.reshape(b, groups, c // groups, h, w, kh, kw)
    patches = patches.permute(0, 1, 3, 4, 2, 5, 6).reshape(b, groups, h*w, -1).long()
    kernels = weight.reshape(groups, co // groups, -1).transpose(1, 2).long()
    result = torch.matmul(patches, kernels).reshape(b, groups, h, w, co // groups)
    result = result.permute(0, 1, 4, 2, 3).reshape(b, co, h, w)
    result = divide_round(result, WEIGHT_SCALE)
    if bias is not None:
        result += bias.long()[None, :, None, None]
    return saturate(result)


def conv2d(x, weight, bias=None, stride=(1, 1), padding=(0, 0), groups=1):
    if x.device.type == 'cpu':
        return conv2d_reference(x, weight, bias, stride, padding, groups)
    from uf_int16_cuda import conv2d as native_conv
    return native_conv(x, weight, bias, *stride, *padding, groups)
