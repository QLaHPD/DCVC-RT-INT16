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
    supports_byte_input = True
    supports_yuv420_input = True
    supports_partial_output = True
    supports_skipped_output = True

    def __init__(self, prepared, device='cuda:0', reconstruct_on_encode=True):
        from src.models.entropy_models import EntropyCoder
        self.variant = prepared['variant']
        self.reconstruct_on_encode = reconstruct_on_encode
        self.gpu_gate = None
        self.device = torch.device(device)
        self.model = make_model(self.variant, meta=True)
        self.layers = Layers(self.model, prepared, self.device)
        self.coder = EntropyCoder()
        self.coder.set_entropy_coder_parallel(1)
        self._masked_decode = getattr(self.coder.decoder, 'decode_y_with_skip', None)
        self._encode_chunk_to_bytes = getattr(self.coder.encoder, 'encode_chunk_to_bytes', None)
        values = prepared['tensors']
        for index, kind in enumerate(('z', 'y')):
            self.coder.set_cdf(values[f'tables.{kind}_cdf'].numpy(),
                               values[f'tables.{kind}_lengths'].numpy(), index)
        if not hasattr(self.coder.decoder, 'get_decoded_tensor'):
            raise RuntimeError('Rebuild UF entropy extension: integer decoder needs get_decoded_tensor')
        self._position_shape = None
        self._positions = None
        self.skip_threshold = 0
        self._prior_native = None
        self._d2h_stream = None
        self._prior_done = None
        if self.device.type == 'cuda':
            self._d2h_stream = torch.cuda.Stream(device=self.device)
            self._prior_done = torch.cuda.Event()
            import uf_int16_cuda
            if all(hasattr(uf_int16_cuda, name) for name in
                   ('prior_encode', 'prior_indexes', 'prior_restore')):
                self._prior_native = uf_int16_cuda
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
            # Persistent workers see unrelated video sizes. The upstream mask
            # dictionary otherwise keeps a GPU allocation for every size seen.
            m.masks.clear()
            masks = (m.get_mask_2x(*shape, self.device) if self.variant == 'ld'
                     else m.get_mask_4x(*shape, self.device))
            # Boolean indexing resolves nonzero positions and synchronizes on
            # every use. Retain only one shape's ordered flat indexes.
            self._positions = tuple(mask.reshape(-1).nonzero().flatten() for mask in masks)
            self._position_shape = tuple(shape)
        return self._positions

    def _pinned_host_buffer(self, attr, shape, dtype):
        buf = getattr(self, attr, None)
        if buf is None or buf.shape != tuple(shape) or buf.dtype != dtype:
            try:
                buf = torch.empty(shape, dtype=dtype, pin_memory=(self.device.type == 'cuda'))
            except RuntimeError:
                buf = torch.empty(shape, dtype=dtype)
            setattr(self, attr, buf)
        return buf

    def _prior(self, y, params, qp, image, encode):
        m, run = self.model, self.layers
        native = self._prior_native
        has_scale_ops = native is not None and hasattr(native, 'prior_scale')
        if image:
            scales, means = params.chunk(2, dim=1)
            q_enc, q_dec = run.parameter('q_scale_y_enc', qp), run.parameter('q_scale_y_dec', qp)
            if encode:
                y = ops.multiply(y, q_enc)
        else:
            q_dec, scales, means = params.chunk(3, dim=1)
            if has_scale_ops:
                if encode:
                    y = native.prior_scale(y, q_dec)
            else:
                q_dec = q_dec.clamp_min(ops.FEATURE_SCALE//2)
                if encode:
                    q_enc = ops.reciprocal(q_dec)
                    y = ops.multiply(y, q_enc)
        shape = scales.shape
        positions = self._mask_positions(shape)
        reduced = run(m.y_spatial_prior_reduction, params) if self.variant != 'ld' else None
        if (native is not None and self.variant != 'ld' and shape[0] == 1
                and reduced.shape == shape):
            adaptor_shape = (1, shape[1] * 2, shape[2], shape[3])
            adaptor_buf = getattr(self, '_adaptor_buf', None)
            if adaptor_buf is None or adaptor_buf.shape != adaptor_shape:
                adaptor_buf = torch.empty(adaptor_shape, dtype=torch.int16, device=self.device)
                self._adaptor_buf = adaptor_buf
            restored = adaptor_buf[:, :shape[1]]
            restored.zero_()
            adaptor_buf[:, shape[1]:].copy_(reduced)
        else:
            adaptor_buf = None
            restored = torch.zeros(shape, dtype=torch.int16, device=self.device)
        if native is not None and encode:
            total_positions = sum(p.numel() for p in positions)
            packed_all = getattr(self, '_packed_all', None)
            if packed_all is None or packed_all.numel() != total_positions:
                packed_all = torch.empty(total_positions, dtype=torch.int32, device=self.device)
                self._packed_all = packed_all
            offset = 0
        else:
            packed_all = None
            pending = []
        pack_adaptor = getattr(run, '_native_packed_block', None) is not None
        for part, position in enumerate(positions):
            if part:
                if self.variant == 'ld':
                    means = run(m.y_spatial_prior, restored, params)
                else:
                    adaptor = getattr(m, f'y_spatial_prior_adaptor_{part}')
                    if adaptor_buf is not None and pack_adaptor:
                        adapted, packed_a = run(adaptor, adaptor_buf, _pack_output=True)
                        prior_out = run(m.y_spatial_prior, adapted, _packed_input=packed_a)
                    elif adaptor_buf is not None:
                        prior_out = run(m.y_spatial_prior, run(adaptor, adaptor_buf))
                    elif image or self.variant == 'htl':
                        prior_out = run(m.y_spatial_prior, run(adaptor, torch.cat((restored, reduced), dim=1)))
                    else:
                        prior_out = run(m.y_spatial_prior, run(adaptor, restored, reduced))
                    if image or self.variant == 'htl':
                        scales, means = prior_out.chunk(2, dim=1)
                    else:
                        means = prior_out
            if native is not None:
                table = run.values['tables.scale_index']
                if encode:
                    count = position.numel()
                    native.prior_encode(y, scales, means, restored, position, table,
                                        self.skip_threshold, packed_all[offset:offset + count])
                    offset += count
                else:
                    # One fixed-size transfer contains both indexes and skip flags.
                    indexes = native.prior_indexes(scales, position, table,
                                                    self.skip_threshold).cpu().numpy()
                    masked_decode = self._masked_decode
                    if masked_decode is not None:
                        symbols = masked_decode(indexes)
                    elif self.skip_threshold:
                        active = indexes >= 0
                        symbols = np.zeros(indexes.size, dtype=np.int8)
                        if active.any():
                            self.coder.decoder.decode_y(indexes[active].astype(np.uint8))
                            symbols[active] = self.coder.decoder.get_decoded_tensor()
                    else:
                        self.coder.decoder.decode_y(indexes.astype(np.uint8))
                        symbols = self.coder.decoder.get_decoded_tensor()
                    native.prior_restore(torch.from_numpy(symbols).to(self.device),
                                         means, restored, position)
                continue
            selected_scales = scales.reshape(-1).index_select(0, position)
            selected_means = means.reshape(-1).index_select(0, position).long()
            if self.skip_threshold:
                active = selected_scales > self.skip_threshold
                has_active = bool(active.any())
                index = (self._indexes(selected_scales[active]) if has_active else
                         torch.empty(0, dtype=torch.uint8, device=self.device))
                symbols = torch.zeros(position.numel(), dtype=torch.int8, device=self.device)
                if encode:
                    residual = y.reshape(-1).index_select(0, position).long() - selected_means
                    symbols[active] = ops.divide_round(residual[active], ops.FEATURE_SCALE).clamp(
                        -128, 127).to(torch.int8)
                    coded_symbols = symbols[active]
                elif has_active:
                    self.coder.decoder.decode_y(index.cpu().numpy())
                    decoded = torch.from_numpy(self.coder.decoder.get_decoded_tensor()).to(self.device)
                    symbols[active] = decoded
            else:
                index = self._indexes(selected_scales)
                if encode:
                    residual = y.reshape(-1).index_select(0, position).long() - selected_means
                    symbols = ops.divide_round(residual, ops.FEATURE_SCALE).clamp(
                        -128, 127).to(torch.int8)
                    coded_symbols = symbols
                else:
                    self.coder.decoder.decode_y(index.cpu().numpy())
                    symbols = torch.from_numpy(self.coder.decoder.get_decoded_tensor()).to(self.device)
            if encode:
                packed = (coded_symbols.to(torch.int16) << 8) | index.to(torch.int16)
                pending.append(packed.cpu().numpy().copy())
            restored.view(-1).index_copy_(0, position,
                ops.saturate(symbols.long()*ops.FEATURE_SCALE + selected_means))
        y_hat = (native.prior_unscale(restored, q_dec)
                 if (not image and has_scale_ops)
                 else ops.multiply(restored, q_dec))
        if native is not None and encode:
            # Defer CPU transfer/filtering so _reconstruct can be launched on the GPU
            # before CPU partition filtering and rANS task submission.
            return y_hat, (packed_all, positions)
        return y_hat, pending

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

    def _reconstruct(self, y_hat, qp, reset, output_pixels=True, output_frames=None):
        m, run = self.model, self.layers
        if self.variant == 'image':
            return run(m.dec, y_hat, run.parameter('q_scale_dec', qp))
        if output_frames == 0:
            output_pixels = False
        feature = run(m.decoder, y_hat, self.context, run.parameter('q_decoder', qp))
        # P-frame encoding only needs decoded features for its next context.
        # A reset additionally needs the final frame in unshuffled form.
        partial = output_frames is not None and output_frames < (1 if self.variant == 'ld' else 8)
        options = {'output_frames': output_frames} if partial else {}
        frames = run(m.recon_head, feature, **options) if output_pixels else None
        self.reference = feature
        if reset:
            if output_pixels and not partial:
                # The reset reference is exactly the last frame before its
                # pixel shuffle. Reuse it instead of running that head twice.
                last = frames[-1] if isinstance(frames, list) else frames
                reference = F.pixel_unshuffle(last, 8)
            else:
                reference = run(m.recon_head, feature, for_reset=True)
            self.clear_dpb()
            self.reference = reference
        return frames

    @torch.no_grad()
    def compress(self, x, qp, *args):
        height = x.shape[3] * 2 if x.dim() == 5 else x.shape[-2]
        if self.gpu_gate is None or self.device.type != 'cuda' or min(height, x.shape[-1]) < 144:
            return self._compress(x, qp, *args)
        with self.gpu_gate:
            try:
                defer = getattr(self, '_encode_chunk_to_bytes', None) is not None
                result = self._compress(x, qp, *args, _defer_entropy=defer) if defer else self._compress(x, qp, *args)
            finally:
                # Release the run's GPU slot only after queued reconstruction.
                torch.cuda.synchronize(self.device)
        entropy_args = result.pop('_entropy_args', None) if isinstance(result, dict) else None
        if entropy_args is not None:
            result['bit_stream'] = self._encode_chunk_to_bytes(*entropy_args)
        return result

    def _compress(self, x, qp, *args, _defer_entropy=False):
        image = self.variant == 'image'
        reset = False if image else args[0]
        if x.dtype != torch.uint8:
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
        encode_chunk_to_bytes = getattr(self, '_encode_chunk_to_bytes', None)
        if isinstance(parts, tuple) and encode_chunk_to_bytes is not None:
            packed_gpu, _ = parts
            if self._d2h_stream is not None:
                self._prior_done.record()
                frames = self._reconstruct(y_hat, qp, reset, self.reconstruct_on_encode)
                packed_cpu = self._pinned_host_buffer('_packed_cpu', packed_gpu.shape, packed_gpu.dtype)
                z_cpu_t = self._pinned_host_buffer('_z_cpu', z.shape, z.dtype)
                with torch.cuda.stream(self._d2h_stream):
                    self._d2h_stream.wait_event(self._prior_done)
                    packed_gpu.record_stream(self._d2h_stream)
                    z.record_stream(self._d2h_stream)
                    packed_cpu.copy_(packed_gpu, non_blocking=True)
                    z_cpu_t.copy_(z, non_blocking=True)
                if not _defer_entropy:
                    self._d2h_stream.synchronize()
                packed = packed_cpu.numpy()
                z_cpu = z_cpu_t.numpy()
            else:
                packed = packed_gpu.cpu().numpy()
                z_cpu = z.cpu().numpy()
                frames = self._reconstruct(y_hat, qp, reset, self.reconstruct_on_encode)
            if _defer_entropy:
                return {'bit_stream': None, 'x_hat': frames, 'ec_parallel': 1,
                        '_entropy_args': (packed, z_cpu, qp * m.z_channel, m.z_channel)}
            bits = encode_chunk_to_bytes(packed, z_cpu, qp * m.z_channel, m.z_channel)
            return {'bit_stream': bits, 'x_hat': frames, 'ec_parallel': 1}
        z_nhwc = z.permute(0, 2, 3, 1).contiguous()
        if isinstance(parts, tuple):
            packed_gpu, positions = parts
            if self._d2h_stream is not None:
                self._prior_done.record()
                # Queue GPU reconstruction on the compute stream before D2H wait.
                frames = self._reconstruct(y_hat, qp, reset, self.reconstruct_on_encode)
                with torch.cuda.stream(self._d2h_stream):
                    self._d2h_stream.wait_event(self._prior_done)
                    packed_gpu.record_stream(self._d2h_stream)
                    z_nhwc.record_stream(self._d2h_stream)
                    packed = packed_gpu.cpu().numpy()
                    z_cpu = z_nhwc.cpu().numpy()
            else:
                packed = packed_gpu.cpu().numpy()
                z_cpu = z_nhwc.cpu().numpy()
                frames = self._reconstruct(y_hat, qp, reset, self.reconstruct_on_encode)
            parts = []
            offset = 0
            for position in positions:
                part = packed[offset:offset + position.numel()]
                parts.append((part[part != 65536] if self.skip_threshold else part).astype(np.int16))
                offset += position.numel()
        else:
            z_cpu = z_nhwc.cpu().numpy()
            frames = self._reconstruct(y_hat, qp, reset, self.reconstruct_on_encode)
        self.coder.encoder.reset()
        # rANS is a stack: decode z first, then spatial partitions in ascending order.
        for part in reversed(parts):
            if part.size:
                self.coder.encoder.encode_y(part)
        self.coder.encoder.encode_z(z_cpu, qp*m.z_channel, m.z_channel)
        self.coder.encoder.flush()
        bits = self.coder.encoder.get_encoded_stream().tobytes()
        return {'bit_stream': bits, 'x_hat': frames, 'ec_parallel': 1}

    @torch.no_grad()
    def decompress(self, bits, sps, qp, ec_part, reset=False, *, output_frames=None):
        options = {'output_frames': output_frames} if output_frames is not None else {}
        if (self.gpu_gate is None or self.device.type != 'cuda'
                or min(sps['width'], sps['height']) < 144):
            return self._decompress(bits, sps, qp, ec_part, reset, **options)
        with self.gpu_gate:
            try:
                return self._decompress(bits, sps, qp, ec_part, reset, **options)
            finally:
                torch.cuda.synchronize(self.device)

    def _decompress(self, bits, sps, qp, ec_part, reset=False, *, output_frames=None):
        if ec_part != 1:
            raise ValueError('Unsupported UF integer entropy parallelism')
        image = self.variant == 'image'
        maximum = 1 if image or self.variant == 'ld' else 8
        minimum = 1 if image else 0
        if output_frames is not None and (type(output_frames) is not int or not minimum <= output_frames <= maximum):
            raise ValueError('Invalid UF reconstruction frame count')
        if not image:
            self._context()
        height, width = (sps['height']+15)//16, (sps['width']+15)//16
        self.coder.decoder.set_stream(np.frombuffer(bits, dtype=np.uint8))
        z = self._decode_z((height+3)//4, (width+3)//4, qp)
        params = self._params(z, qp, height, width)
        y_hat, _ = self._prior(None, params, qp, image, encode=False)
        frames = self._reconstruct(y_hat, qp, reset, output_frames=output_frames)
        return {'x_hat': [] if output_frames == 0 else frames}
