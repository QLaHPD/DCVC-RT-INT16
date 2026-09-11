"""Differentiable DCVC-RT graphs over the deployment model's parameters."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from src.models.architecture import ImageArchitecture
from src.training.config import LossConfig
from src.training.ops import TrainingOps
from src.utils.transforms import ycbcr2rgb


QP_OFFSETS = (0, 8, 0, 4, 0, 4, 0, 4)


def pad_latent(y):
    return F.pad(y, (0, (-y.shape[-1]) % 4, 0, (-y.shape[-2]) % 4), mode="replicate")


def summed_bits(probability):
    return -torch.log2(probability.clamp_min(1e-9)).flatten(1).sum(1)


class TrainingModel(nn.Module):
    """Wrap an unfused codec without changing its checkpoint state keys."""

    def __init__(self, codec, mode="float", activation_checkpointing=False, force_zero_thres=None):
        super().__init__()
        self.codec = codec
        self.kind = "image" if isinstance(codec.architecture, ImageArchitecture) else "video"
        self.ops = TrainingOps(mode, activation_checkpointing, force_zero_thres)

    def rate_z(self, z, symbols, qp):
        index = torch.full((z.shape[0],), qp, dtype=torch.int32, device=z.device)
        value = z + torch.empty_like(z).uniform_(-0.5, 0.5) if self.training and not self.ops.quantized else symbols
        estimator = self.codec.bit_estimator_z
        return summed_bits(estimator(value + 0.5, index) - estimator(value - 0.5, index))

    def rate_y(self, residual, symbols, scales, mask):
        value = residual + torch.empty_like(residual).uniform_(-0.5, 0.5) \
            if self.training and not self.ops.quantized else symbols
        scale = self.ops.gaussian_scale(scales, self.codec.gaussian_encoder)
        # erfc on abs(symbol) avoids subtraction of two CDFs both close to 1.
        centered = value.abs()
        inv = 1 / (scale * math.sqrt(2))
        probability = 0.5 * (torch.erfc((centered - 0.5) * inv)
                             - torch.erfc((centered + 0.5) * inv))
        active = mask.bool()
        if self.ops.force_zero_thres is not None:
            active = active & (scales > self.ops.force_zero_thres)
        return summed_bits(torch.where(active, probability, torch.ones_like(probability)))

    def prior(self, y, params):
        net, op = self.codec, self.ops
        if self.kind == "image":
            q_enc, q_dec = op.prior_quant(params[:, :2]).chunk(2, 1)
            scales, means = params[:, 2:].chunk(2, 1)
            common = op.apply(net.y_spatial_prior_reduction, params)
            masks = net.get_mask_4x(*y.shape, y.dtype, y.device)
            adaptors = [net.y_spatial_prior_adaptor_1, net.y_spatial_prior_adaptor_2,
                        net.y_spatial_prior_adaptor_3]
            spatial = net.y_spatial_prior
        else:
            q, scales, means = params.chunk(3, 1)
            q_enc, q_dec = op.reciprocal(q)
            common = params
            masks = net.get_mask_2x(*y.shape, y.dtype, y.device)
            adaptors = [None]
            spatial = net.y_spatial_prior.conv
        y = op.mul(y, q_enc)
        total_bits = torch.zeros(y.shape[0], device=y.device)
        restored = None
        all_symbols, all_scales = [], []
        for step, mask in enumerate(masks):
            if step:
                context = torch.cat((restored, common), dim=1)
                if adaptors[step - 1] is not None:
                    context = op.apply(adaptors[step - 1], context)
                scales, means = op.apply(spatial, context).chunk(2, 1)
            residual, symbols, part, masked_scales = op.masked(y, scales, means, mask)
            total_bits = total_bits + self.rate_y(residual, symbols, scales, mask)
            restored = part if restored is None else op.add(restored, part)
            all_symbols.append(symbols)
            all_scales.append(masked_scales)
        return op.mul(restored, q_dec), total_bits, all_symbols, all_scales

    def image(self, x, qp):
        net, op = self.codec, self.ops
        quant_enc = op.parameter(net.q_scale_enc[qp:qp + 1])
        quant_dec = op.parameter(net.q_scale_dec[qp:qp + 1])
        x = op.input(x)
        feature = op.apply(net.enc.enc_1, F.pixel_unshuffle(x, 8))
        y = op.apply(net.enc.enc_2, op.mul(feature, quant_enc))
        z = op.apply(net.hyper_enc, pad_latent(y))
        z_hat, z_symbols = op.symbols_z(z)
        params = op.apply(net.y_prior_fusion, op.apply(net.hyper_dec, z_hat))
        params = params[:, :, :y.shape[2], :y.shape[3]].contiguous()
        y_hat, bits_y, symbols, scales = self.prior(y, params)
        out = op.apply(net.dec.dec_1, y_hat)
        out = op.apply(net.dec.dec_2, op.mul(out, quant_dec))
        x_hat = F.pixel_shuffle(out, 8).clamp(0, 1)
        return {"x_hat": x_hat, "bits_y": bits_y, "bits_z": self.rate_z(z, z_symbols, qp),
                "y_symbols": symbols, "y_scales": scales, "z_symbols": z_symbols,
                "feature": None}

    def video(self, x, qp, reference, refresh=False):
        net, op = self.codec, self.ops
        x = op.input(x)
        q_enc = op.parameter(net.q_encoder[qp:qp + 1])
        q_dec = op.parameter(net.q_decoder[qp:qp + 1])
        q_feat = op.parameter(net.q_feature[qp:qp + 1])
        q_recon = op.parameter(net.q_recon[qp:qp + 1])
        if reference.get("feature") is None or refresh:
            # Reference reconstructions already live on the fixed-point grid;
            # do not reapply the media input's half-precision conversion.
            frame = reference["x_hat"]
            feature = op.apply(net.feature_adaptor_i, F.pixel_unshuffle(frame, 8))
        else:
            feature = op.apply(net.feature_adaptor_p, reference["feature"])
        context = op.apply(net.feature_extractor.conv1, feature)
        context_t = op.mul(context, q_feat)
        context = op.apply(net.feature_extractor.conv2, context)
        feature = op.fused_encoder_adaptor(net.encoder, F.pixel_unshuffle(x, 8), context)
        feature = op.apply(net.encoder.conv3, feature)
        y = op.conv(op.mul(feature, q_enc), net.encoder.down)
        z = op.apply(net.hyper_encoder.conv, pad_latent(y))
        z_hat, z_symbols = op.symbols_z(z)
        hyper = op.apply(net.hyper_decoder.conv, z_hat)
        temporal = op.apply(net.temporal_prior_encoder, context_t)
        hyper = hyper[:, :, :temporal.shape[2], :temporal.shape[3]].contiguous()
        params = op.apply(net.y_prior_fusion.conv, torch.cat((hyper, temporal), dim=1))
        y_hat, bits_y, symbols, scales = self.prior(y, params)
        feature = op.apply(net.decoder.up, y_hat)
        feature = op.apply(net.decoder.conv1, torch.cat((feature, context), dim=1))
        feature = op.conv_post(feature, net.decoder.conv2, quant=q_dec)
        out = op.apply(net.recon_generation_net.conv, feature)
        out = op.mul(out, q_recon)
        x_hat = op.conv_post(out, net.recon_generation_net.head, shuffle=8)
        return {"x_hat": x_hat, "feature": feature, "bits_y": bits_y,
                "bits_z": self.rate_z(z, z_symbols, qp), "z_symbols": z_symbols,
                "y_symbols": symbols, "y_scales": scales}

    def forward(self, x, qp, reference=None, refresh=False):
        if type(qp) is not int or not 0 <= qp < (64 if self.kind == "image" else 72):
            raise ValueError("QP is outside this model's quality bank")
        if x.ndim != 4 or x.shape[1] != 3 or x.shape[-1] % 16 or x.shape[-2] % 16:
            raise ValueError("training input must be [batch, 3, height, width], padded to multiples of 16")
        if self.kind == "image":
            return self.image(x, qp)
        if reference is None:
            raise ValueError("video forward requires a reconstructed reference")
        return self.video(x, qp, reference, refresh)


def distortion(x, x_hat, config=None):
    config = LossConfig() if config is None else config
    mse = (x - x_hat).square().mean(dim=(-2, -1))
    weights = x.new_tensor(config.yuv_channel_weights)
    yuv = (mse * weights).sum(1) / weights.sum()
    rgb = (ycbcr2rgb(x, clamp=False) - ycbcr2rgb(x_hat, clamp=False)).square().mean(dim=(1, 2, 3))
    return config.yuv_weight * yuv + (1 - config.yuv_weight) * rgb


def rd_loss(x, result, qp, kind, config=None, frame_index=0):
    config = LossConfig() if config is None else config
    endpoints = config.image_lambdas if kind == "image" else config.video_lambdas
    lam = math.exp(math.log(endpoints[0]) + qp / 63 * math.log(endpoints[1] / endpoints[0]))
    d = distortion(x, result["x_hat"], config)
    pixels = x.shape[-1] * x.shape[-2]
    bpp_y, bpp_z = result["bits_y"] / pixels, result["bits_z"] / pixels
    weight = 1.0 if kind == "image" else config.frame_weights[frame_index % 8]
    loss = (bpp_y + bpp_z + lam * weight * d).mean()
    return loss, {"distortion": d.mean().detach(), "bpp_y": bpp_y.mean().detach(),
                  "bpp_z": bpp_z.mean().detach(), "loss": loss.detach()}
