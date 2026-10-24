"""Execute UF layer topology using integer tensors and prepared parameters only."""

import torch
import torch.nn.functional as F
from torch import nn

from . import ops


class Layers:
    supports_byte_input = True
    supports_partial_output = True

    def __init__(self, model, prepared, device):
        self.names = {id(module): name for name, module in model.named_modules()}
        self.values = {name: value.to(device) for name, value in prepared['tensors'].items()
                       if not name.startswith(('tables.z_', 'tables.y_'))}
        self._parameters = {}
        self._fused_ffns = {}
        self._fused_blocks = {}
        self._native_blocks = {}
        self._native_block = None
        self._native_packed_block = None
        self._native_conv_packed = None
        self._packed_chains = {}
        post_native = False
        if torch.device(device).type == 'cuda':
            import uf_int16_cuda
            post_native = callable(getattr(uf_int16_cuda, 'conv2d_post', None))
            self._native_block = getattr(uf_int16_cuda, 'depth_conv_block', None)
            self._native_packed_block = getattr(uf_int16_cuda, 'depth_conv_block_packed', None)
            self._native_conv_packed = getattr(uf_int16_cuda, 'conv2d_packed', None)
        for module in model.modules():
            if (post_native and type(module).__name__ in ('DepthConvBlock', 'SpatialPriorAdaptor')
                    and len(module.dc) == 4 and len(module.ffn) == 3
                    and type(module.dc[1]).__name__ == 'WSiLU'
                    and type(module.ffn[1]).__name__ == 'WSiLUChunkAdd'):
                convolutions = (module.dc[0], module.dc[3], module.ffn[0], module.ffn[2])
                if all(isinstance(conv, nn.Conv2d) and conv.kernel_size == (1, 1)
                       and conv.stride == (1, 1) and conv.padding == (0, 0)
                       and conv.dilation == (1, 1) and conv.groups == 1
                       and conv.padding_mode == 'zeros' for conv in convolutions):
                    params = []
                    for conv in convolutions:
                        prefix = self.names[id(conv)]
                        params.extend((self.values[prefix+'.weight'], self.values.get(prefix+'.bias')))
                    self._fused_blocks[id(module)] = (module.dc[2], *params)
                    depth = module.dc[2]
                    if (callable(self._native_block) and isinstance(depth, nn.Conv2d)
                            and depth.kernel_size == (3, 3) and depth.stride == (1, 1)
                            and depth.padding == (1, 1) and depth.dilation == (1, 1)
                            and depth.groups == depth.in_channels == depth.out_channels
                            and depth.padding_mode == 'zeros'):
                        blocks = (module.dc[0], depth, module.dc[3], module.ffn[0], module.ffn[2])
                        prefixes = [self.names[id(conv)] for conv in blocks]
                        self._native_blocks[id(module)] = (
                            [self.values[prefix+'.weight'] for prefix in prefixes],
                            [self.values.get(prefix+'.bias') for prefix in prefixes], module.shortcut)
            if not isinstance(module, nn.Sequential) or len(module) != 3:
                continue
            conv = module[0]
            if (not isinstance(conv, nn.Conv2d) or type(module[1]).__name__ != 'WSiLUChunkAdd'
                    or conv.kernel_size != (1, 1) or conv.stride != (1, 1)
                    or conv.padding != (0, 0) or conv.dilation != (1, 1)
                    or conv.groups != 1 or conv.padding_mode != 'zeros'):
                continue
            prefix = self.names[id(conv)]
            self._fused_ffns[id(module)] = (self.values[prefix+'.weight'],
                                           self.values.get(prefix+'.bias'), module[2])

        if callable(self._native_packed_block):
            for module in model.modules():
                if not isinstance(module, nn.Sequential):
                    continue
                children = list(module)
                entries = []
                for index, child in enumerate(children):
                    current = self._native_blocks.get(id(child))
                    eligible, channels = False, 0
                    if current is not None and index + 1 < len(children):
                        weights = current[0]
                        channels, depth = weights[2].shape[0], weights[1].shape[0]
                        if (channels >= 256 and channels <= 32768 and channels % 32 == 0
                                and depth >= 256 and depth % 32 == 0):
                            nxt = children[index + 1]
                            following = self._native_blocks.get(id(nxt))
                            if following is not None:
                                if nxt.adaptor is None:
                                    eligible = following[0][0].shape[1] == channels
                                elif (callable(self._native_conv_packed)
                                      and isinstance(nxt.adaptor, nn.Conv2d)
                                      and nxt.adaptor.kernel_size == (1, 1)
                                      and nxt.adaptor.stride == (1, 1)
                                      and nxt.adaptor.padding == (0, 0)
                                      and nxt.adaptor.dilation == (1, 1)
                                      and nxt.adaptor.groups == 1
                                      and nxt.adaptor.padding_mode == 'zeros'):
                                    eligible = nxt.adaptor.in_channels == channels
                            elif (callable(self._native_conv_packed)
                                  and isinstance(nxt, nn.Conv2d)
                                  and nxt.kernel_size == (1, 1)
                                  and nxt.stride == (1, 1)
                                  and nxt.padding == (0, 0)
                                  and nxt.dilation == (1, 1)
                                  and nxt.groups == 1
                                  and nxt.padding_mode == 'zeros'):
                                eligible = nxt.in_channels == channels
                    entries.append((eligible, channels))
                if any(eligible for eligible, _ in entries):
                    self._packed_chains[id(module)] = entries

    def parameter(self, name, qp):
        key = (name, qp)
        if key not in self._parameters:
            self._parameters[key] = self.values[name][qp:qp+1, :, None, None]
        return self._parameters[key]

    def __call__(self, module, *args, for_reset=False, output_frames=None,
                 _packed_input=None, _pack_output=False):
        x = args[0]
        name = type(module).__name__
        source_bytes = x.dtype != torch.int16
        if source_bytes and (x.dtype != torch.uint8 or not x.is_cuda or
                             name not in ('Encoder', 'IntraEncoder')):
            raise TypeError(f'Integer layer received {x.dtype}')
        if isinstance(module, nn.Sequential):
            fused = self._fused_ffns.get(id(module))
            if fused is not None:
                weight, bias, projection = fused
                return self(projection, ops.conv2d_wsilu4(x, weight, bias, self.values['tables.wsilu']))
            chain = self._packed_chains.get(id(module))
            if chain is not None and self._native_packed_block is not None and x.shape[2]*x.shape[3] >= 128:
                packed = _packed_input
                for child, (eligible, channels) in zip(module, chain):
                    emit = eligible and x.shape[0]*channels*x.shape[2]*x.shape[3] <= 8*1024*1024
                    result = self(child, x, _packed_input=packed, _pack_output=emit)
                    if emit:
                        x, packed = result
                    else:
                        x, packed = result, None
                return x
            for child in module:
                x = self(child, x)
            return x
        if isinstance(module, nn.Conv2d):
            if module.dilation != (1, 1) or module.padding_mode != 'zeros':
                raise ValueError('Unsupported integer convolution configuration')
            prefix = self.names[id(module)]
            w, b = self.values[prefix+'.weight'], self.values.get(prefix+'.bias')
            if (self._native_conv_packed is not None and (_pack_output or _packed_input is not None)
                    and module.kernel_size == (1, 1) and module.stride == (1, 1)
                    and module.padding == (0, 0) and module.groups == 1):
                out, packed = self._native_conv_packed(x, w, b, _packed_input, _pack_output)
                return (out, packed) if _pack_output else out
            out = ops.conv2d(x, w, b, module.stride, module.padding, module.groups)
            return (out, None) if _pack_output else out
        if isinstance(module, nn.PixelShuffle):
            return F.pixel_shuffle(x, module.upscale_factor)
        if name == 'WSiLU':
            return ops.lookup(x, self.values['tables.wsilu'])
        if name == 'WSiLUChunkAdd':
            return ops.wsilu4(x, self.values['tables.wsilu'])
        if name in ('DepthConvBlock', 'SpatialPriorAdaptor'):
            if len(args) == 2:
                x = torch.cat(args, dim=1)
                _packed_input = None
            native = self._native_blocks.get(id(module))
            if module.adaptor is not None:
                if native is not None and self._native_packed_block is not None and self._native_conv_packed is not None:
                    x, _packed_input = self(module.adaptor, x, _packed_input=_packed_input, _pack_output=True)
                else:
                    x = self(module.adaptor, x)
                    _packed_input = None
            if native is not None:
                weights, biases, shortcut = native
                if self._native_packed_block is not None and (_pack_output or _packed_input is not None):
                    out, packed = self._native_packed_block(x, weights, biases,
                        self.values['tables.wsilu'], shortcut, _packed_input, _pack_output)
                    return (out, packed) if _pack_output else out
                return self._native_block(x, weights, biases, self.values['tables.wsilu'], shortcut)
            fused = self._fused_blocks.get(id(module))
            if fused is not None:
                depth, w0, b0, w1, b1, w2, b2, w3, b3 = fused
                lut = self.values['tables.wsilu']
                out = self(depth, ops.conv2d_post(x, w0, b0, lut=lut))
                out = ops.conv2d_post(out, w1, b1, residual=x)
                expanded = ops.conv2d_wsilu4(out, w2, b2, lut)
                return ops.conv2d_post(expanded, w3, b3, residual=out,
                                       shortcut=x if module.shortcut else None)
            out = ops.add(self(module.dc, x), x)
            out = ops.add(self(module.ffn, out), out)
            return ops.add(out, x) if module.shortcut else out
        if name == 'SubpelConv2x':
            return self(module.conv, x)
        if name == 'ResidualBlockUpsample':
            return self(module.conv, self(module.up, x))
        if name == 'ResidualBlockWithStride2':
            unshuffled = F.pixel_unshuffle(x, 2)
            if self._native_packed_block is not None and self._native_conv_packed is not None:
                down, packed = self(module.down, unshuffled, _pack_output=True)
                return self(module.conv, down, _packed_input=packed)
            return self(module.conv, self(module.down, unshuffled))
        if name == 'IntraEncoder':
            source = ops.from_bytes_unshuffle(x) if source_bytes else F.pixel_unshuffle(x, 8)
            return self(module.enc_2, ops.multiply(self(module.enc_1, source), args[1]))
        if name == 'IntraDecoder':
            return F.pixel_shuffle(self(module.dec_2, ops.multiply(self(module.dec_1, x), args[1])), 8)
        if name == 'Encoder':
            source = ops.from_bytes_unshuffle(x) if source_bytes else F.pixel_unshuffle(x, 8)
            out = self(module.conv1, torch.cat((source, args[1]), dim=1))
            if hasattr(module, 'conv2'):
                out = self(module.conv2, out)
            return self(module.down, ops.multiply(out, args[2]))
        if name == 'Decoder':
            out = self(module.conv1, torch.cat((self(module.up, x), args[1]), dim=1))
            if hasattr(module, 'conv2'):
                out = self(module.conv2, out)
            return ops.multiply(out, args[2])
        if name == 'FeatureAdaptorM':
            return self(module.conv, torch.cat(args, dim=1))
        if name == 'TemporalPriorEncoder':
            return self(module.conv, ops.multiply(x, args[1]) if len(args)==2 else x)
        if name == 'PriorFusion':
            other = ops.multiply(args[1], args[2]) if len(args)==3 else args[1]
            return self(module.conv, torch.cat((x, other), dim=1))
        if name == 'ReconHead':
            maximum = 1 if hasattr(module, 'head') else 8
            count = maximum if output_frames is None else output_frames
            if type(count) is not int or not 1 <= count <= maximum:
                raise ValueError('Invalid UF reconstruction frame count')
            if hasattr(module, 'head'):  # LD
                out = self(module.head, self(module.conv, x))
                return out if for_reset else F.pixel_shuffle(out, 8)
            if module.is_hts:
                if for_reset:
                    return self(module.conv2[-1], self(module.conv1[-1], x))
                frames = []
                for index in range(count):
                    if index % 2 == 0:
                        shared = self(module.conv1[index//2], x)
                    frames.append(F.pixel_shuffle(self(module.conv2[index], shared), 8))
                return frames
            if for_reset:
                return self(module.conv[-1], x)
            packed_x = (x.permute(0, 2, 3, 1).contiguous()
                        if (count > 1 and self._native_packed_block is not None
                            and self._native_conv_packed is not None and x.is_cuda
                            and x.is_contiguous() and x.shape[2]*x.shape[3] >= 128
                            and x.numel() <= 8*1024*1024 and not (x.data_ptr() & 7))
                        else None)
            return ([F.pixel_shuffle(self(child, x, _packed_input=packed_x), 8)
                     for child in module.conv[:count]]
                    if packed_x is not None
                    else [F.pixel_shuffle(self(child, x), 8) for child in module.conv[:count]])
        if name in ('FeatureAdaptorI', 'FeatureExtractor', 'HyperEncoder', 'HyperDecoder',
                    'IntraHyperEncoder', 'IntraHyperDecoder', 'IntraSpatialPrior', 'IntraYPriorFusion',
                    'SpatialPrior'):
            if len(args) == 1 and _packed_input is not None:
                return self(module.conv, x, _packed_input=_packed_input)
            return self(module.conv, torch.cat(args, dim=1) if len(args)>1 else x)
        raise NotImplementedError(f'UF integer layer not implemented: {name}')
