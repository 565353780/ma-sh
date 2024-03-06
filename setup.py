import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='mash_cpp',
    ext_modules=[
        CppExtension(
            name='mash_cpp',
            sources=[
                'ma_sh/Cpp/src/add.cpp',
                'ma_sh/Cpp/src/idx.cpp',
                'ma_sh/Cpp/main.cpp',
            ],
            include_dirs = ['ma_sh/Cpp'],
            extra_compile_args=['-g']),
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    })
