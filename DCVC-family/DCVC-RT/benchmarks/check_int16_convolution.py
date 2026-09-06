#!/usr/bin/env python3
"""Full-range, overflow, odd-shape, grouped, and non-default-stream regression cases."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from benchmarks.benchmark_int16_conv import load_extension

p = argparse.ArgumentParser(__doc__)
p.add_argument('--extension', type=Path, required=True)
p.add_argument('--save', type=Path)
p.add_argument('--compare', type=Path)
a = p.parse_args()
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.utils.deterministic.fill_uninitialized_memory = False
ext = load_extension(a.extension)
g = torch.Generator().manual_seed(20260905)
refs = torch.load(a.compare, weights_only=True) if a.compare else None
outputs = {}
shapes = [(1,1,1,1,1), (2,3,5,3,7), (1,17,65,5,13), (2,33,31,7,9),
          (1,64,64,8,8), (1,368,1472,12,22), (1,512,256,12,22),
          (1,128,1024,6,11), (1,256,256,24,40)]
stream = torch.cuda.Stream()
# CPU -> CUDA transfers, dispatch, and consumers all on a non-default stream.
with torch.cuda.stream(stream):
    for kernel in (1, 3):
        for shape in shapes:
            b,c,m,h,w = shape
            for with_bias in (False, True):
                for span in (512,32768):
                    key = f'{shape}_{kernel}_{with_bias}_{span}'
                    x = torch.randint(-span,span,(b,c,h,w),generator=g,dtype=torch.int16).cuda()
                    weight = torch.randint(-span,span,(m,c,kernel,kernel),generator=g,dtype=torch.int16).cuda()
                    bias = torch.randint(-32768,32768,(m,),generator=g,dtype=torch.int16).cuda() if with_bias else None
                    out = ext.conv2d_int16_cuda(x,weight,bias,1,1,kernel//2,kernel//2,1)
                    outputs[key] = out.cpu()
    for terms in (1,2,3,4,7,8,31,32,33,64):
        for xv,wv in ((-32768,-32768),(-32768,32767),(32767,32767),(1,4096),(-1,4096),(1,4095)):
            key = f'overflow_{terms}_{xv}_{wv}'
            x = torch.full((1,terms,1,3),xv,dtype=torch.int16,device='cuda')
            w = torch.full((65,terms,1,1),wv,dtype=torch.int16,device='cuda')
            outputs[key] = ext.conv2d_int16_cuda(x,w,None,1,1,0,0,1).cpu()
    for k,s,pad,groups in ((1,1,1,2),(1,2,0,1),(2,2,0,1),(3,2,1,1),(3,1,1,4),(5,1,2,2)):
        x = torch.randint(-32768,32768,(2,4,9,14),generator=g,dtype=torch.int16).cuda()[...,::2]
        w = torch.randint(-32768,32768,(4,4//groups,k,k),generator=g,dtype=torch.int16).cuda()
        bias = torch.arange(8,dtype=torch.int16,device='cuda')[::2]
        outputs[f'fallback_{k}_{s}_{pad}_{groups}'] = ext.conv2d_int16_cuda(x,w,bias,s,s,pad,pad,groups).cpu()
stream.synchronize()
if refs is not None:
    assert outputs.keys() == refs.keys()
    for name,out in outputs.items():
        if not torch.equal(out,refs[name]):
            mismatch = out != refs[name]
            raise AssertionError(f'{name}: {mismatch.sum().item()} mismatches, got={out[mismatch][:8]}, expected={refs[name][mismatch][:8]}')
    print(f'PASS: all {len(outputs)} convolution cases are bit-exact')
if a.save:
    torch.save(outputs,a.save)
    print(f'Saved {len(outputs)} reference cases to {a.save}')
if hasattr(ext, 'round_and_to_int8_int16_cuda'):
    with torch.cuda.stream(stream):
        values = torch.arange(-32768,32768,device='cuda',dtype=torch.int32).to(torch.int16)
        for x in (values, values.reshape(256,256).t(), values[:0]):
            feature, symbols = ext.round_and_to_int8_int16_cuda(x)
            xi = x.to(torch.int32)
            q = (xi.abs()+256)//512
            q = torch.where(xi < 0,-q,q).clamp(-128,127)
            assert torch.equal(feature,(q*512).clamp(-32768,32767).to(torch.int16))
            assert torch.equal(symbols,q.to(torch.int8))
    stream.synchronize()
    print('PASS: fused rounding matches all 65,536 INT16 values, noncontiguous and empty inputs')
