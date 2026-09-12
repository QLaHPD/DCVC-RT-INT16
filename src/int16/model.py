"""UF integer intra/inter model and rANS coding, with identical decode state."""

import numpy as np
import torch
import torch.nn.functional as F

from . import ops
from .layers import Layers
from .prepared import make_model


def pad_right_bottom(x, multiple):
    h, w = x.shape[-2:]
    bottom, right = (-h) % multiple, (-w) % multiple
    # Index-based replication also works for integer CPU/CUDA tensors.
    if bottom:
        x = torch.cat((x, x[:, :, -1:, :].expand(-1, -1, bottom, -1)), dim=2)
    if right:
        x = torch.cat((x, x[:, :, :, -1:].expand(-1, -1, -1, right)), dim=3)
    return x


class IntegerModel:
    def __init__(self, prepared, device='cuda:0', reconstruct_on_encode=True):
        from src.models.entropy_models import EntropyCoder
        self.variant = prepared['variant']
        self.reconstruct_on_encode = reconstruct_on_encode
        self.device = torch.device(device)
        self.model = make_model(self.variant, meta=True)
        self.layers = Layers(self.model, prepared, self.device)
        self.coder = EntropyCoder()
        self.coder.set_entropy_coder_parallel(1)
        values = prepared['tensors']
        for index, kind in enumerate(('z', 'y')):
            self.coder.set_cdf(values[f'tables.{kind}_cdf'].numpy(),
                               values[f'tables.{kind}_lengths'].numpy(), index)
        if not hasattr(self.coder.decoder, 'get_decoded_tensor'):
            raise RuntimeError('Rebuild UF entropy extension: integer decoder needs get_decoded_tensor')
        self._position_shape = None
        self._positions = None
        self.clear_dpb()

    def clear_dpb(self):
        self.reference = self.memory = self.context = None

    def add_ref_feature_from_frame(self, frame, apply_feature_adaptor=True):
        self.clear_dpb()
        self.reference = F.pixel_unshuffle(frame, 8)

    def _context(self):
        if self.reference is None:
            raise ValueError('Integer P-frame has no reference')
        if self.memory is None:
            self.memory = self.layers(self.model.feature_adaptor_i, self.reference)
        else:
            self.memory = self.layers(self.model.feature_adaptor_m, self.memory, self.reference)
        self.context = self.layers(self.model.feature_extractor, self.memory)

    def _decode_z(self, height, width, qp):
        ch = self.model.z_channel
        self.coder.decoder.decode_z(height*width*ch, qp*ch, ch)
        values = self.coder.decoder.get_decoded_tensor()
        values = torch.from_numpy(values).reshape(1, height, width, ch).permute(0, 3, 1, 2)
        return values.to(self.device)

    def _indexes(self, scales):
        return ops.lookup(scales, self.layers.values['tables.scale_index'])

    def _mask_positions(self, shape):
        if self._position_shape != tuple(shape):
            m = self.model
            masks = (m.get_mask_2x(*shape, self.device) if self.variant == 'ld'
                     else m.get_mask_4x(*shape, self.device))
            # Boolean indexing resolves nonzero positions and synchronizes on
            # every use. Retain only one shape's ordered flat indexes.
            self._positions = tuple(mask.reshape(-1).nonzero().flatten() for mask in masks)
            self._position_shape = tuple(shape)
        return self._positions

    def _prior(self, y, params, qp, image, encode):
        m, run = self.model, self.layers
        if image:
            scales, means = params.chunk(2, dim=1)
            q_enc, q_dec = run.parameter('q_scale_y_enc', qp), run.parameter('q_scale_y_dec', qp)
        else:
            q_dec, scales, means = params.chunk(3, dim=1)
            q_dec = q_dec.clamp_min(ops.FEATURE_SCALE//2)
            q_enc = ops.reciprocal(q_dec)
        if encode:
            y = ops.multiply(y, q_enc)
        shape = scales.shape
        positions = self._mask_positions(shape)
        reduced = run(m.y_spatial_prior_reduction, params) if self.variant != 'ld' else None
        restored = torch.zeros(shape, dtype=torch.int16, device=self.device)
        pending = []
        for part, position in enumerate(positions):
            if part:
                if self.variant == 'ld':
                    means = run(m.y_spatial_prior, restored, params)
                else:
                    adaptor = getattr(m, f'y_spatial_prior_adaptor_{part}')
                    if image or self.variant == 'htl':
                        next_params = run(m.y_spatial_prior, run(adaptor, torch.cat((restored, reduced), dim=1)))
                        scales, means = next_params.chunk(2, dim=1)
                    else:
                        means = run(m.y_spatial_prior, run(adaptor, restored, reduced))
            index = self._indexes(scales.reshape(-1).index_select(0, position))
            selected_means = means.reshape(-1).index_select(0, position).long()
            if encode:
                residual = y.reshape(-1).index_select(0, position).long() - selected_means
                symbols = ops.divide_round(residual, ops.FEATURE_SCALE).clamp(-128, 127).to(torch.int8)
                packed = (symbols.to(torch.int16) << 8) | index.to(torch.int16)
                pending.append(packed.cpu().numpy().copy())
            else:
                self.coder.decoder.decode_y(index.cpu().numpy())
                symbols = torch.from_numpy(self.coder.decoder.get_decoded_tensor()).to(self.device)
            restored.view(-1).index_copy_(0, position,
                ops.saturate(symbols.long()*ops.FEATURE_SCALE + selected_means))
        return ops.multiply(restored, q_dec), pending

    def _params(self, z, qp, height, width):
        m, run = self.model, self.layers
        if self.variant == 'image':
            params = run(m.y_prior_fusion, run(m.hyper_dec, ops.from_symbols(z)))
            return params[:, :, :height, :width]
        hyper = run(m.hyper_decoder, ops.from_symbols(z))[:, :, :height, :width]
        quant = run.parameter('q_feature', qp)
        if self.variant == 'ld':
            temporal = run(m.temporal_prior_encoder, self.memory)
            return run(m.y_prior_fusion, hyper, temporal, quant)
        temporal = run(m.temporal_prior_encoder, self.memory, quant)
        return run(m.y_prior_fusion, hyper, temporal)

    def _reconstruct(self, y_hat, qp, reset, output_pixels=True):
        m, run = self.model, self.layers
        if self.variant == 'image':
            return run(m.dec, y_hat, run.parameter('q_scale_dec', qp))
        feature = run(m.decoder, y_hat, self.context, run.parameter('q_decoder', qp))
        # P-frame encoding only needs decoded features for its next context.
        # A reset additionally needs the final frame in unshuffled form.
        frames = run(m.recon_head, feature) if output_pixels else None
        self.reference = feature
        if reset:
            reference = run(m.recon_head, feature, for_reset=True)
            self.clear_dpb()
            self.reference = reference
        return frames

    @torch.no_grad()
    def compress(self, x, qp, *args):
        image = self.variant == 'image'
        reset = False if image else args[0]
        x = pad_right_bottom(x, 16)
        m, run = self.model, self.layers
        if image:
            y = run(m.enc, x, run.parameter('q_scale_enc', qp))
            z = ops.symbols(run(m.hyper_enc, pad_right_bottom(y, 4)))
        else:
            self._context()
            y = run(m.encoder, x, self.context, run.parameter('q_encoder', qp))
            z = ops.symbols(run(m.hyper_encoder, pad_right_bottom(y, 4)))
        params = self._params(z, qp, *y.shape[-2:])
        y_hat, parts = self._prior(y, params, qp, image, encode=True)
        self.coder.encoder.reset()
        # rANS is a stack: decode z first, then spatial partitions in ascending order.
        for part in reversed(parts):
            self.coder.encoder.encode_y(part)
        self.coder.encoder.encode_z(z.permute(0,2,3,1).contiguous().cpu().numpy(),
                                    qp*m.z_channel, m.z_channel)
        self.coder.encoder.flush()
        bits = self.coder.encoder.get_encoded_stream().tobytes()
        frames = self._reconstruct(y_hat, qp, reset, self.reconstruct_on_encode)
        return {'bit_stream': bits, 'x_hat': frames, 'ec_parallel': 1}

    @torch.no_grad()
    def decompress(self, bits, sps, qp, ec_part, reset=False):
        if ec_part != 1:
            raise ValueError('Unsupported UF integer entropy parallelism')
        image = self.variant == 'image'
        if not image:
            self._context()
        height, width = (sps['height']+15)//16, (sps['width']+15)//16
        self.coder.decoder.set_stream(np.frombuffer(bits, dtype=np.uint8))
        z = self._decode_z((height+3)//4, (width+3)//4, qp)
        params = self._params(z, qp, height, width)
        y_hat, _ = self._prior(None, params, qp, image, encode=False)
        return {'x_hat': self._reconstruct(y_hat, qp, reset)}
