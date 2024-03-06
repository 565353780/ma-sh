import torch
from setuptools import setup
from torch.utils.cpp_extension import (
    CUDAExtension, CppExtension, BuildExtension
)

sources = [
    'ma_sh/Cpp/src/add.cpp',
    'ma_sh/Cpp/src/idx.cpp',
    'ma_sh/Cpp/main.cpp',
]

include_dirs = ['ma_sh/Cpp']

if torch.cuda.is_available():
    module = CUDAExtension(
        name = "mash_cpp",
        sources = sources,
        include_dirs = include_dirs,
    )
else:
    module = CppExtension(
        name = "mash_cpp",
        sources = sources,
        include_dirs = include_dirs,
    )

setup(
    name = "MASH-CPP",
    version = "0.0.1",
    ext_modules = [module],
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)}
)
