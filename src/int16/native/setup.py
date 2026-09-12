import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ.setdefault('MAX_JOBS', '1')
setup(name='uf_int16_cuda', ext_modules=[CUDAExtension(
    'uf_int16_cuda', ['bind.cpp', 'conv.cu', 'elementwise.cu'],
    depends=['int16_mma.cuh'],
    extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']},
)], cmdclass={'build_ext': BuildExtension})
