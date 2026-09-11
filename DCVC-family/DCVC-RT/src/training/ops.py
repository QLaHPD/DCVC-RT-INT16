"""Training operations with exact deployed INT16 values and surrogate gradients.

Inference modules are deliberately not called here: they cache detached weights,
use kernels without backwards, and may fuse parameters in-place. Parameter
quantization in this module is recomputed on every forward, including checkpoint
recomputation. No optimizer update can leave stale prepared weights behind.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from src.layers import int16_inference as integer
from src.layers import cuda_inference as cuda
from src.layers.layers import (DepthConvBlock, ResidualBlockUpsample,
                               ResidualBlockWithStride2, SubpelConv2x, WSiLU, WSiLUChunkAdd)


class ExactValue(torch.autograd.Function):
    """Return exact values without cancellation in x + (exact - x).detach()."""

    @staticmethod
    def forward(ctx, surrogate, exact):
        return exact.clone()

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def round_ste(x, minimum=-128.0, maximum=127.0):
    proxy = x.clamp(minimum, maximum)
    return ExactValue.apply(proxy, x.detach().round().clamp(minimum, maximum))


class TrainingOps:
    def __init__(self, mode="float", activation_checkpointing=False, force_zero_thres=None):
        if mode not in {"float", "int16"}:
            raise ValueError(f"unsupported training arithmetic: {mode}")
        if mode == "int16" and not integer.CUSTOMIZED_INT16_CUDA_INFERENCE:
            raise RuntimeError("exact INT16 training requires ./build_native_extensions.sh on this machine")
        self.mode = mode
        self.quantized = mode == "int16"
        self.activation_checkpointing = activation_checkpointing
        self.force_zero_thres = force_zero_thres
        self.saturated = None
        self.elements = 0

    @staticmethod
    def native(x):
        if not x.is_cuda:
            raise RuntimeError("exact INT16 forward requires a CUDA tensor")
        return integer.feature_to_int16(x.detach())

    def finish(self, surrogate, exact, minimum=-64.0, maximum=32767 / 512):
        if exact.dtype == torch.int16:
            # Keep the counters on-device: no per-layer host synchronization.
            count = ((exact == -32768) | (exact == 32767)).sum().detach()
            self.saturated = count if self.saturated is None else self.saturated + count
            self.elements += exact.numel()
            exact = integer.feature_to_float(exact)
        return ExactValue.apply(surrogate.clamp(minimum, maximum), exact.detach())

    def parameter(self, tensor, scale=512):
        if not self.quantized or tensor is None:
            return tensor
        lower, upper = -32768 / scale, 32767 / scale
        exact = tensor.detach().float().mul(scale).round().clamp(-32768, 32767) / scale
        return ExactValue.apply(tensor.clamp(lower, upper), exact)

    def input(self, x):
        # The deployed media input paths convert their normalized tensors to
        # half before feature quantization. Preserve that boundary under QAT.
        if not self.quantized:
            return x
        return self.finish(x, self.native(x.half().float()))

    def conv(self, x, module, weight=None, bias=True):
        weight = module.weight if weight is None else weight
        bias = module.bias if bias is True else bias
        w, b = self.parameter(weight, 8192), self.parameter(bias)
        surrogate = F.conv2d(x, w, b, module.stride, module.padding, module.dilation, module.groups)
        if not self.quantized:
            return surrogate
        if module.dilation != (1, 1):
            raise ValueError("INT16 training does not support dilated convolutions")
        exact = integer.conv2d_int16_cuda(
            self.native(x), integer.weight_to_int16(weight.detach()),
            None if bias is None else integer.bias_to_int16(bias.detach()),
            *module.stride, *module.padding, module.groups)
        return self.finish(surrogate, exact)

    def add(self, x, y):
        proxy = x + y
        if not self.quantized:
            return proxy
        return self.finish(proxy, integer.add_tensors_int16(self.native(x), self.native(y)))

    def mul(self, x, y):
        y = self.parameter(y)
        proxy = x * y
        if not self.quantized:
            return proxy
        return self.finish(proxy, integer.mul_feature_scale_int16(self.native(x), self.native(y)))

    def wsilu(self, x):
        proxy = x * torch.sigmoid(4 * x)
        if not self.quantized:
            return proxy
        return self.finish(proxy, integer.wsilu_int16(self.native(x)))

    def prior_quant(self, x):
        proxy = torch.sigmoid(x) * 1.5 + 0.5
        if not self.quantized:
            return proxy
        return self.finish(proxy, integer.prior_quant_step_int16(self.native(x)))

    def reciprocal(self, x):
        q_dec = x.clamp_min(0.5)
        proxy = q_dec.reciprocal()
        if not self.quantized:
            return proxy, q_dec
        exact_dec, exact_enc = integer.reciprocal_scale_int16(self.native(x), 0.5)
        return self.finish(proxy, exact_enc), self.finish(q_dec, exact_dec)

    def conv_post(self, x, module, *, shuffle=None, quant=None):
        """Preserve pre-bias saturation and the fused bias/scale operation."""
        kernel = (integer.bias_quant_int16_cuda if quant is not None else
                  integer.bias_pixel_shuffle_2_int16_cuda if shuffle == 2 else
                  integer.bias_pixel_shuffle_8_int16_cuda)
        if not self.quantized or module.bias is None or kernel is None:
            out = self.conv(x, module)
            if quant is not None:
                out = self.mul(out, quant)
            if shuffle is not None:
                out = F.pixel_shuffle(out, shuffle)
            return out.clamp(0, 1) if shuffle == 8 else out
        out = self.conv(x, module, bias=None)
        bias = self.parameter(module.bias)
        proxy = out + bias[None, :, None, None]
        native_bias = integer.bias_to_int16(module.bias.detach())
        if quant is not None:
            quant = self.parameter(quant)
            proxy = proxy * quant
            exact = kernel(self.native(out), native_bias, self.native(quant))
        else:
            proxy = F.pixel_shuffle(proxy, shuffle)
            exact = kernel(self.native(out), native_bias)
        if shuffle == 8:
            proxy = proxy.clamp(0, 1)
        return self.finish(proxy, exact)

    def depth(self, module, x, *, adapted=False):
        if module.adaptor is not None and not adapted:
            x = self.conv(x, module.adaptor)
        out = self.conv(x, module.dc[0])
        out = self.wsilu(out)
        out = self.conv(out, module.dc[2])
        out = self.add(self.conv(out, module.dc[3]), x)
        identity = self.add(out, x) if module.shortcut and self.quantized else out
        ffn = self.conv(out, module.ffn[0])
        first, second = self.wsilu(ffn).chunk(2, 1)
        ffn = self.add(first, second)
        out = self.add(self.conv(ffn, module.ffn[2]), identity)
        if module.shortcut and not self.quantized:
            out = out + x
        return out

    def apply(self, module, x):
        if isinstance(module, nn.Sequential):
            for child in module:
                x = self.apply(child, x)
            return x
        if isinstance(module, nn.Conv2d):
            return self.conv(x, module)
        if isinstance(module, DepthConvBlock):
            if self.activation_checkpointing and torch.is_grad_enabled():
                return checkpoint(lambda value: self.depth(module, value), x,
                                  use_reentrant=False, preserve_rng_state=True)
            return self.depth(module, x)
        if isinstance(module, ResidualBlockWithStride2):
            return self.apply(module.conv, self.conv(x, module.down))
        if isinstance(module, ResidualBlockUpsample):
            return self.apply(module.conv, self.apply(module.up, x))
        if isinstance(module, SubpelConv2x):
            return self.conv_post(x, module.conv[0], shuffle=2)
        if isinstance(module, WSiLU):
            return self.wsilu(x)
        if isinstance(module, WSiLUChunkAdd):
            first, second = self.wsilu(x).chunk(2, 1)
            return self.add(first, second)
        if isinstance(module, nn.PixelShuffle):
            return module(x)
        raise TypeError(f"unsupported training module: {type(module).__name__}")

    def fused_encoder_adaptor(self, encoder, x, ctx):
        """Canonical CPU fusion, differentiable back to unfused GPU weights.

        Inference prepares this matmul on CPU before moving to CUDA. Doing it on
        CUDA during QAT can round fused weights differently near a half step.
        CPU copies remain connected to autograd; no .data mutation is used.
        """
        conv, adaptor = encoder.conv1, encoder.conv2[0].adaptor
        if encoder.fuse_conv1_flag:
            raise ValueError("training requires unfused encoder parameters")
        if not self.quantized:
            return self.apply(encoder.conv2, torch.cat((self.conv(x, conv), ctx), dim=1))
        channels = encoder.channels
        a, w = adaptor.weight.float().cpu(), conv.weight.float().cpu()
        a_bias, b = adaptor.bias.float().cpu(), conv.bias.float().cpu()
        left = a[:, :channels, 0, 0]
        fused_weight = torch.cat(((left @ w[:, :, 0, 0])[:, :, None, None], a[:, channels:]), dim=1)
        fused_bias = a_bias + (left @ b[:, None])[:, 0]
        out = self.conv(torch.cat((x, ctx), dim=1), adaptor,
                        weight=fused_weight.to(x.device), bias=fused_bias.to(x.device))
        out = self.depth(encoder.conv2[0], out, adapted=True)
        for block in encoder.conv2[1:]:
            out = self.apply(block, out)
        return out

    def symbols_z(self, z):
        symbols = round_ste(z)
        if not self.quantized:
            return symbols, symbols
        native_symbols, _ = integer.round_feature_to_symbols(self.native(z))
        symbols = ExactValue.apply(z.clamp(-128, 127), native_symbols.float())
        z_hat = self.finish(symbols, integer.symbol_to_feature(native_symbols))
        return z_hat, symbols

    def masked(self, y, scales, means, mask):
        residual = (y - means) * mask
        symbols = round_ste(residual)
        if self.force_zero_thres is not None:
            symbols = symbols * (scales > self.force_zero_thres)
        reconstructed = (symbols + means) * mask
        if self.quantized:
            # Mask values are 0/1, not fixed-point features.
            mask_int = mask.to(dtype=torch.int16).contiguous()
            thres = -1 if self.force_zero_thres is None else round(self.force_zero_thres * 512)
            kernel = cuda.process_with_mask_int16_cuda
            if kernel is not None:
                _, exact_symbols, exact_hat, _ = kernel(
                    self.native(y), self.native(scales), self.native(means), mask_int, thres)
            else:
                active = mask_int != 0
                int_res = torch.where(active, self.native(y).int() - self.native(means).int(), 0)
                exact_symbols = integer.round_divide(int_res, 512).clamp(-128, 127).short()
                if self.force_zero_thres is not None:
                    exact_symbols = torch.where(self.native(scales).int() > thres, exact_symbols, 0)
                exact_hat = integer.clip_to_int16(exact_symbols.int() * 512 + self.native(means).int() * mask_int)
            symbols = ExactValue.apply(symbols, exact_symbols.float())
            reconstructed = self.finish(reconstructed, exact_hat)
        return residual, symbols, reconstructed, scales * mask

    def gaussian_scale(self, scales, gaussian):
        proxy = scales.clamp(gaussian.scale_min, gaussian.scale_max)
        if not self.quantized:
            return proxy
        index = integer.scale_to_index_int16(self.native(scales), gaussian.scale_min,
                                            gaussian.scale_max, gaussian.log_scale_min,
                                            gaussian.log_step_recip)
        exact = gaussian.scale_table.to(scales.device)[index.long()]
        return ExactValue.apply(proxy, exact)
