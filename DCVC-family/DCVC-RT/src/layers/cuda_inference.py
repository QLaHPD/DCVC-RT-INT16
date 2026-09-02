# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

import torch
import torch.nn.functional as F

from .extension_loader import load_inference_extensions
from .int16_inference import FEATURE_SCALE, add_bias_int16, add_tensors_int16, bias_to_int16, \
    clip_to_int16, clamp_quant_step_int16, feature_to_int16, get_scale_index_lut, \
    int16_inference_enabled, mul_feature_scale_int16, reciprocal_scale_int16, round_divide, \
    round_feature_to_symbols, scale_to_index_int16, symbol_to_feature


CUSTOMIZED_CUDA_INFERENCE = False
process_with_mask_int16_cuda = None
combine_for_reading_int16_cuda = None
restore_y_parts_int16_cuda = None
build_index_dec_int16_cuda = None
build_index_enc_int16_cuda = None
add_and_multiply_int16_cuda = None
bias_quant_int16_cuda = None
bias_pixel_shuffle_8_int16_cuda = None
try:
    _ext = load_inference_extensions(
        required=(
            "process_with_mask_cuda",
            "combine_for_reading_2x_cuda",
            "restore_y_2x_cuda",
            "restore_y_4x_cuda",
            "build_index_dec_cuda",
            "round_and_to_int8_cuda",
            "clamp_reciprocal_with_quant_cuda",
            "bias_quant_cuda",
            "add_and_multiply_cuda",
            "bias_pixel_shuffle_8_cuda",
            "replicate_pad_cuda",
            "build_index_enc_cuda",
            "DepthConvProxy",
            "SubpelConv2xProxy",
        )
    )
    process_with_mask_cuda = _ext.process_with_mask_cuda
    combine_for_reading_2x_cuda = _ext.combine_for_reading_2x_cuda
    restore_y_2x_cuda = _ext.restore_y_2x_cuda
    restore_y_4x_cuda = _ext.restore_y_4x_cuda
    build_index_dec_cuda = _ext.build_index_dec_cuda
    round_and_to_int8_cuda = _ext.round_and_to_int8_cuda
    clamp_reciprocal_with_quant_cuda = _ext.clamp_reciprocal_with_quant_cuda
    bias_quant_cuda = _ext.bias_quant_cuda
    add_and_multiply_cuda = _ext.add_and_multiply_cuda
    bias_pixel_shuffle_8_cuda = _ext.bias_pixel_shuffle_8_cuda
    replicate_pad_cuda = _ext.replicate_pad_cuda
    build_index_enc_cuda = _ext.build_index_enc_cuda
    DepthConvProxy = _ext.DepthConvProxy
    SubpelConv2xProxy = _ext.SubpelConv2xProxy
    process_with_mask_int16_cuda = getattr(_ext, "process_with_mask_int16_cuda", None)
    combine_for_reading_int16_cuda = getattr(_ext, "combine_for_reading_int16_cuda", None)
    restore_y_parts_int16_cuda = getattr(_ext, "restore_y_parts_int16_cuda", None)
    build_index_dec_int16_cuda = getattr(_ext, "build_index_dec_int16_cuda", None)
    build_index_enc_int16_cuda = getattr(_ext, "build_index_enc_int16_cuda", None)
    add_and_multiply_int16_cuda = getattr(_ext, "add_and_multiply_int16_cuda", None)
    bias_quant_int16_cuda = getattr(_ext, "bias_quant_int16_cuda", None)
    bias_pixel_shuffle_8_int16_cuda = getattr(_ext, "bias_pixel_shuffle_8_int16_cuda", None)
    CUSTOMIZED_CUDA_INFERENCE = True
except Exception:  # pylint: disable=W0718
    pass


if not CUSTOMIZED_CUDA_INFERENCE and 'SUPPRESS_CUSTOM_KERNEL_WARNING' not in os.environ:
    print("cannot import cuda implementation for inference, fallback to pytorch.")


def round_and_to_int8(z):
    if int16_inference_enabled() and z.dtype == torch.int16 and z.is_cuda:
        z_q, z_int8 = round_feature_to_symbols(z)
        z_hat = symbol_to_feature(z_q)
        return z_hat, z_int8
    if CUSTOMIZED_CUDA_INFERENCE and z.is_cuda:
        z_int8 = round_and_to_int8_cuda(z)
        return z, z_int8

    z_hat = torch.clamp(torch.round(z), -128., 127.)
    z_hat_write = z_hat.to(dtype=torch.int8)
    return z_hat, z_hat_write


def clamp_reciprocal_with_quant(q_dec, y, min_val):
    if int16_inference_enabled() and q_dec.dtype == torch.int16 and q_dec.is_cuda:
        q_dec_clamp, q_enc = reciprocal_scale_int16(q_dec, min_val)
        y = mul_feature_scale_int16(y, q_enc)
        return q_dec_clamp, y
    if CUSTOMIZED_CUDA_INFERENCE and q_dec.is_cuda:
        # q_dec is not inplace modified at decoder side
        q_dec = clamp_reciprocal_with_quant_cuda(q_dec, y, min_val)
        return q_dec, y

    q_dec = torch.clamp_min(q_dec, min_val)
    q_enc = torch.reciprocal(q_dec)
    y = y * q_enc
    return q_dec, y


def add_and_multiply(y_hat_0, y_hat_1, q_dec):
    if int16_inference_enabled() and y_hat_0.dtype == torch.int16 and y_hat_0.is_cuda:
        if add_and_multiply_int16_cuda is not None:
            return add_and_multiply_int16_cuda(y_hat_0.contiguous(), y_hat_1.contiguous(),
                                               q_dec.contiguous())
        return mul_feature_scale_int16(add_tensors_int16(y_hat_0, y_hat_1), q_dec)
    if CUSTOMIZED_CUDA_INFERENCE and y_hat_0.is_cuda:
        add_and_multiply_cuda(y_hat_0, y_hat_1, q_dec)
        return y_hat_0

    y_hat = y_hat_0 + y_hat_1
    y_hat = y_hat * q_dec
    return y_hat


def process_with_mask(y, scales, means, mask, force_zero_thres):
    if int16_inference_enabled() and y.dtype == torch.int16 and y.is_cuda:
        if process_with_mask_int16_cuda is not None:
            thres = -1 if force_zero_thres is None else int(round(force_zero_thres * FEATURE_SCALE))
            return process_with_mask_int16_cuda(y.contiguous(), scales.contiguous(),
                                                means.contiguous(), mask.contiguous(), thres)
        mask_bool = mask != 0
        zeros = torch.zeros_like(y)
        scales_hat = torch.where(mask_bool, scales, zeros)
        means_hat = torch.where(mask_bool, means, zeros)
        y_res = torch.where(mask_bool, y.to(dtype=torch.int32) - means.to(dtype=torch.int32),
                            torch.zeros_like(y, dtype=torch.int32))
        y_q = round_divide(y_res, FEATURE_SCALE).clamp_(-128, 127).to(dtype=torch.int16)
        if force_zero_thres is not None:
            thres = int(round(force_zero_thres * FEATURE_SCALE))
            y_q = torch.where(scales_hat.to(dtype=torch.int32) > thres, y_q, torch.zeros_like(y_q))
        y_hat = clip_to_int16(y_q.to(dtype=torch.int32) * FEATURE_SCALE + means_hat.to(dtype=torch.int32))
        return clip_to_int16(y_res), y_q, y_hat, scales_hat
    if CUSTOMIZED_CUDA_INFERENCE and y.is_cuda:
        thres = force_zero_thres if force_zero_thres is not None else -1.
        return process_with_mask_cuda(y, scales, means, mask, thres)

    scales_hat = scales * mask
    means_hat = means * mask

    y_res = (y - means_hat) * mask
    y_q = torch.round(y_res)
    if force_zero_thres is not None:
        cond = scales_hat > force_zero_thres
        y_q = y_q * cond
    y_q = torch.clamp(y_q, -128., 127.)
    y_hat = y_q + means_hat

    return y_res, y_q, y_hat, scales_hat


def combine_for_reading_2x(x, mask, inplace=False):
    if int16_inference_enabled() and x.dtype == torch.int16 and x.is_cuda:
        if combine_for_reading_int16_cuda is not None:
            if inplace:
                out = x[:, :x.shape[1] // 2, :, :]
            else:
                out = torch.empty((x.shape[0], x.shape[1] // 2, x.shape[2], x.shape[3]),
                                  dtype=x.dtype, layout=x.layout, device=x.device)
            combine_for_reading_int16_cuda(out, x.contiguous(), mask.contiguous(), 2)
            return out
        x_masked = torch.where(mask != 0, x, torch.zeros_like(x))
        x0, x1 = x_masked.chunk(2, 1)
        out_val = clip_to_int16(x0.to(dtype=torch.int32) + x1.to(dtype=torch.int32))
        if inplace:
            out = x[:, :x.shape[1] // 2, :, :]
            out.copy_(out_val)
            return out
        return out_val
    if CUSTOMIZED_CUDA_INFERENCE and x.is_cuda and x.is_contiguous():
        B, C, H, W = x.shape
        if inplace:
            out = x[:, :C // 2, :, :]
        else:
            out = torch.empty((B, C // 2, H, W), dtype=x.dtype, layout=x.layout, device=x.device)
        combine_for_reading_2x_cuda(out, x, mask)
        return out

    x = x * mask
    x0, x1 = x.chunk(2, 1)
    return x0 + x1


def combine_for_reading_4x(x, mask):
    if int16_inference_enabled() and x.dtype == torch.int16 and x.is_cuda:
        if combine_for_reading_int16_cuda is not None:
            out = torch.empty((x.shape[0], x.shape[1] // 4, x.shape[2], x.shape[3]),
                              dtype=x.dtype, layout=x.layout, device=x.device)
            combine_for_reading_int16_cuda(out, x.contiguous(), mask.contiguous(), 4)
            return out
        x_masked = torch.where(mask != 0, x, torch.zeros_like(x))
        x0, x1, x2, x3 = x_masked.chunk(4, 1)
        return clip_to_int16((x0.to(dtype=torch.int32) + x1.to(dtype=torch.int32)) +
                             (x2.to(dtype=torch.int32) + x3.to(dtype=torch.int32)))
    x = x * mask
    x0, x1, x2, x3 = x.chunk(4, 1)
    return (x0 + x1) + (x2 + x3)


def restore_y_2x(y, means, mask):
    if int16_inference_enabled() and means.dtype == torch.int16 and means.is_cuda:
        if restore_y_parts_int16_cuda is not None:
            out = torch.empty_like(means)
            restore_y_parts_int16_cuda(out, y.contiguous(), means.contiguous(), mask.contiguous(), 2)
            return out
        y_feature = symbol_to_feature(y)
        base = torch.cat((y_feature, y_feature), dim=1)
        out = clip_to_int16(base.to(dtype=torch.int32) + means.to(dtype=torch.int32))
        return torch.where(mask != 0, out, torch.zeros_like(out))
    if CUSTOMIZED_CUDA_INFERENCE and y.is_cuda and y.is_contiguous():
        out = torch.empty_like(means)
        restore_y_2x_cuda(out, y, means, mask)
        return out

    return (torch.cat((y, y), dim=1) + means) * mask


def restore_y_2x_with_cat_after(y, means, mask, to_cat):
    if int16_inference_enabled() and means.dtype == torch.int16 and means.is_cuda:
        out = restore_y_2x(y, means, mask)
        return out, torch.cat((out, to_cat), dim=1)
    if CUSTOMIZED_CUDA_INFERENCE and y.is_cuda and y.is_contiguous():
        B, C1, H, W = means.shape
        C2 = to_cat.shape[1]
        out = torch.empty((B, C1 + C2, H, W), dtype=means.dtype, layout=means.layout,
                          device=means.device)
        restore_y_2x_cuda(out[:, :C1, :, :], y, means, mask)
        out[:, C1:, :, :] = to_cat
        return out[:, :C1, :, :], out

    out = (torch.cat((y, y), dim=1) + means) * mask
    return out, torch.cat((out, to_cat), dim=1)


def restore_y_4x(y, means, mask):
    if int16_inference_enabled() and means.dtype == torch.int16 and means.is_cuda:
        if restore_y_parts_int16_cuda is not None:
            out = torch.empty_like(means)
            restore_y_parts_int16_cuda(out, y.contiguous(), means.contiguous(), mask.contiguous(), 4)
            return out
        y_feature = symbol_to_feature(y)
        base = torch.cat((y_feature, y_feature, y_feature, y_feature), dim=1)
        out = clip_to_int16(base.to(dtype=torch.int32) + means.to(dtype=torch.int32))
        return torch.where(mask != 0, out, torch.zeros_like(out))
    if CUSTOMIZED_CUDA_INFERENCE and y.is_cuda and y.is_contiguous():
        out = torch.empty_like(means)
        restore_y_4x_cuda(out, y, means, mask)
        return out

    return (torch.cat((y, y, y, y), dim=1) + means) * mask


def build_index_dec(scales, scale_min, scale_max, log_scale_min, log_step_recip, skip_thres=None):
    if int16_inference_enabled() and scales.dtype == torch.int16 and scales.is_cuda:
        if build_index_dec_int16_cuda is not None:
            lut = get_scale_index_lut(scales.device, scale_min, scale_max, log_scale_min, log_step_recip)
            thres = -1 if skip_thres is None else int(round(skip_thres * FEATURE_SCALE))
            out, skip_cond = build_index_dec_int16_cuda(scales.contiguous(), lut, thres)
            return out, None if skip_cond.numel() == 0 else skip_cond
        out = scale_to_index_int16(scales, scale_min, scale_max, log_scale_min, log_step_recip)
        skip_cond = None
        if skip_thres is not None:
            thres = int(round(skip_thres * FEATURE_SCALE))
            skip_cond = scales.to(dtype=torch.int32) > thres
        return out, skip_cond
    if CUSTOMIZED_CUDA_INFERENCE and scales.is_cuda:
        out = torch.empty_like(scales, dtype=torch.uint8)
        skip_cond = None
        if skip_thres is not None:
            skip_cond = torch.empty_like(scales, dtype=torch.bool)
        else:
            skip_thres = -1.

        build_index_dec_cuda(out, skip_cond, scales, scale_min, scale_max, log_scale_min,
                             log_step_recip, skip_thres)
        return out, skip_cond

    skip_cond = None
    scales = scales.clamp_(scale_min, scale_max)
    indexes = (torch.log(scales) - log_scale_min) * log_step_recip
    indexes = indexes.to(dtype=torch.uint8)
    if skip_thres is not None:
        skip_cond = scales > skip_thres
    return indexes, skip_cond


def build_index_enc(symbols, scales, scale_min, scale_max, log_scale_min,
                    log_step_recip, skip_thres=None):
    if int16_inference_enabled() and scales.dtype == torch.int16 and scales.is_cuda:
        if build_index_enc_int16_cuda is not None:
            lut = get_scale_index_lut(scales.device, scale_min, scale_max, log_scale_min, log_step_recip)
            thres = -1 if skip_thres is None else int(round(skip_thres * FEATURE_SCALE))
            return build_index_enc_int16_cuda(symbols.to(dtype=torch.int16).contiguous(),
                                              scales.contiguous(), lut, thres)
        indexes = scale_to_index_int16(scales, scale_min, scale_max, log_scale_min, log_step_recip)
        out = (symbols.to(dtype=torch.int16) << 8) + indexes.to(dtype=torch.int16)
        if skip_thres is not None:
            thres = int(round(skip_thres * FEATURE_SCALE))
            out = out[scales.to(dtype=torch.int32) > thres]
        return out
    if CUSTOMIZED_CUDA_INFERENCE and scales.is_cuda:
        out = torch.empty_like(scales, dtype=torch.int16)
        skip_cond = None
        if skip_thres is not None:
            skip_cond = torch.empty_like(scales, dtype=torch.bool)
        else:
            skip_thres = -1.

        build_index_enc_cuda(out, skip_cond, symbols, scales, scale_min, scale_max, log_scale_min,
                             log_step_recip, skip_thres)

        out = out[skip_cond]
        return out

    scales = scales.clamp_(scale_min, scale_max)
    indexes = (torch.log(scales) - log_scale_min) * log_step_recip
    indexes = indexes.to(dtype=torch.uint8)
    symbols = symbols.to(dtype=torch.int16)
    out = (symbols << 8) + indexes
    out = out.to(dtype=torch.int16)
    if skip_thres is not None:
        skip_cond = scales > skip_thres
        out = out[skip_cond]
    return out


def replicate_pad(x, pad_b, pad_r):
    if pad_b == 0 and pad_r == 0:
        return x
    if CUSTOMIZED_CUDA_INFERENCE and x.is_cuda:
        return replicate_pad_cuda(x, pad_b, pad_r)
    return F.pad(x, (0, pad_r, 0, pad_b), mode="replicate")


def bias_pixel_shuffle_8(x, bias):
    if int16_inference_enabled() and x.dtype == torch.int16 and x.is_cuda:
        if bias_pixel_shuffle_8_int16_cuda is not None:
            return bias_pixel_shuffle_8_int16_cuda(
                x.contiguous(), bias_to_int16(bias).to(device=x.device).contiguous()
            )
        out = add_bias_int16(x, bias)
        out = F.pixel_shuffle(out, 8)
        return torch.clamp(out.to(dtype=torch.int32), 0, FEATURE_SCALE).to(dtype=torch.int16)
    if CUSTOMIZED_CUDA_INFERENCE and x.is_cuda:
        B, C, H, W = x.shape
        assert B == 1
        out = torch.empty((B, 3, H * 8, W * 8), dtype=x.dtype, device=x.device, layout=x.layout)
        bias_pixel_shuffle_8_cuda(out, x, bias, C, H * W, W, True)
        return out

    out = x + bias[None, :, None, None]
    out = F.pixel_shuffle(out, 8)
    out = torch.clamp(out, 0., 1.)
    return out


def bias_quant(x, bias, quant_step):
    if int16_inference_enabled() and x.dtype == torch.int16 and x.is_cuda:
        if bias_quant_int16_cuda is not None:
            return bias_quant_int16_cuda(
                x.contiguous(),
                bias_to_int16(bias).to(device=x.device).contiguous(),
                feature_to_int16(quant_step).to(device=x.device).contiguous(),
            )
        return mul_feature_scale_int16(add_bias_int16(x, bias), quant_step)
    if CUSTOMIZED_CUDA_INFERENCE and x.is_cuda:
        bias_quant_cuda(x, bias, quant_step)
        return x

    out = x + bias[None, :, None, None]
    out = out * quant_step
    return out
