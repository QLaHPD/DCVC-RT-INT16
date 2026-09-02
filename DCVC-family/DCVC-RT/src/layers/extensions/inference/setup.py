# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import glob
import platform
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


cxx_flags = ["-O3"]
nvcc_flags = ["-O3", "--use_fast_math", "--extra-device-vectorization"]
if sys.platform == 'win32':
    cxx_flags = ["/O2"]


# Let PyTorch manage CUDA arch flags unless the caller pins them explicitly.
# Jetson builds are particularly sensitive to hard-coded nvcc arch values.
if not os.environ.get("TORCH_CUDA_ARCH_LIST") and platform.machine() == "aarch64":
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.7+PTX"


setup(
    name='inference_extensions_cuda',
    ext_modules=[
        CUDAExtension(
            name='inference_extensions_cuda',
            sources=glob.glob('*.cpp') + glob.glob('*.cu'),
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
