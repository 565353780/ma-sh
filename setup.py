import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import (
    CUDAExtension, CppExtension, BuildExtension
)

root_path = os.getcwd() + '/../ma-sh/ma_sh/Cpp/'

sources = [
    root_path + 'src/add.cpp',
    root_path + 'src/idx.cpp',
    root_path + 'src/mask.cpp',
    root_path + 'src/sample.cpp',
    root_path + 'src/filter.cpp',
    root_path + 'main.cpp',
]

include_dirs = [root_path + 'include']

extra_compile_args=[
    '-std=c++23',
    '-DCMAKE_BUILD_TYPE Release'
    '-O3',
    '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
    '-D_GLIBCXX_USE_CXX11_ABI=0',
]

if torch.cuda.is_available():
    module = CUDAExtension(
        name = "mash_cpp",
        sources = sources,
        include_dirs = include_dirs,
        extra_compile_args=extra_compile_args,
    )
else:
    module = CppExtension(
        name = "mash_cpp",
        sources = sources,
        include_dirs = include_dirs,
        extra_compile_args=extra_compile_args,
    )

setup(
    name = "MASH-CPP",
    version = "0.0.1",
    ext_modules = [module],
    cmdclass={'build_ext': BuildExtension}
)
