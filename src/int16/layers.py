"""Execute UF layer topology using integer tensors and prepared parameters only."""

import torch
import torch.nn.functional as F
from torch import nn

from . import ops


class Layers:
    def __init__(self, model, prepared, device):
        self.names = {id(module): name for name, module in model.named_modules()}
        self.values = {name: value.to(device) for name, value in prepared['tensors'].items()
                       if not name.startswith(('tables.z_', 'tables.y_'))}
        self._fused_ffns = {}
        self._fused_blocks = {}
        self._native_blocks = {}
        self._native_block = None
        post_native = False
        if torch.device(device).type == 'cuda':
            import uf_int16_cuda
            post_native = callable(getattr(uf_int16_cuda, 'conv2d_post', None))
            self._native_block = getattr(uf_int16_cuda, 'depth_conv_block', None)
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

    def parameter(self, name, qp):
        return self.values[name][qp:qp+1, :, None, None]

    def __call__(self, module, *args, for_reset=False):
        x = args[0]
        if x.dtype != torch.int16:
            raise TypeError(f'Integer layer received {x.dtype}')
        name = type(module).__name__
        if isinstance(module, nn.Sequential):
            fused = self._fused_ffns.get(id(module))
            if fused is not None:
                weight, bias, projection = fused
                return self(projection, ops.conv2d_wsilu4(x, weight, bias, self.values['tables.wsilu']))
            for child in module:
                x = self(child, x)
            return x
        if isinstance(module, nn.Conv2d):
            if module.dilation != (1, 1) or module.padding_mode != 'zeros':
                raise ValueError('Unsupported integer convolution configuration')
            prefix = self.names[id(module)]
            return ops.conv2d(x, self.values[prefix+'.weight'], self.values.get(prefix+'.bias'),
                              module.stride, module.padding, module.groups)
        if isinstance(module, nn.PixelShuffle):
            return F.pixel_shuffle(x, module.upscale_factor)
        if name == 'WSiLU':
            return ops.lookup(x, self.values['tables.wsilu'])
        if name == 'WSiLUChunkAdd':
            return ops.wsilu4(x, self.values['tables.wsilu'])
        if name in ('DepthConvBlock', 'SpatialPriorAdaptor'):
            if len(args) == 2:
                x = torch.cat(args, dim=1)
            if module.adaptor is not None:
                x = self(module.adaptor, x)
            native = self._native_blocks.get(id(module))
            if native is not None:
                weights, biases, shortcut = native
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
            return self(module.conv, self(module.down, F.pixel_unshuffle(x, 2)))
        if name == 'IntraEncoder':
            return self(module.enc_2, ops.multiply(self(module.enc_1, F.pixel_unshuffle(x, 8)), args[1]))
        if name == 'IntraDecoder':
            return F.pixel_shuffle(self(module.dec_2, ops.multiply(self(module.dec_1, x), args[1])), 8)
        if name == 'Encoder':
            out = self(module.conv1, torch.cat((F.pixel_unshuffle(x, 8), args[1]), dim=1))
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
            if hasattr(module, 'head'):  # LD
                out = self(module.head, self(module.conv, x))
                return out if for_reset else F.pixel_shuffle(out, 8)
            if module.is_hts:
                if for_reset:
                    return self(module.conv2[-1], self(module.conv1[-1], x))
                frames = []
                for index in range(8):
                    if index % 2 == 0:
                        shared = self(module.conv1[index//2], x)
                    frames.append(F.pixel_shuffle(self(module.conv2[index], shared), 8))
                return frames
            if for_reset:
                return self(module.conv[-1], x)
            return [F.pixel_shuffle(self(child, x), 8) for child in module.conv]
        if name in ('FeatureAdaptorI', 'FeatureExtractor', 'HyperEncoder', 'HyperDecoder',
                    'IntraHyperEncoder', 'IntraHyperDecoder', 'IntraSpatialPrior', 'IntraYPriorFusion',
                    'SpatialPrior'):
            return self(module.conv, torch.cat(args, dim=1) if len(args)>1 else x)
        raise NotImplementedError(f'UF integer layer not implemented: {name}')
