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

    def parameter(self, name, qp):
        return self.values[name][qp:qp+1, :, None, None]

    def __call__(self, module, *args, for_reset=False):
        x = args[0]
        if x.dtype != torch.int16:
            raise TypeError(f'Integer layer received {x.dtype}')
        name = type(module).__name__
        if isinstance(module, nn.Sequential):
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
