import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, CppExtension, BuildExtension

root_path = os.getcwd() + "/../ma-sh/ma_sh/Cpp/"
src_path = root_path + "src/"

sources = [
    src_path + "filter.cpp",
    src_path + "idx.cpp",
    src_path + "inv.cpp",
    src_path + "mash.cpp",
    src_path + "mash_unit.cpp",
    src_path + "mask.cpp",
    src_path + "rotate.cpp",
    src_path + "sample.cpp",
    src_path + "sh.cpp",
    src_path + "value.cpp",
    root_path + "main.cpp",
]

include_dirs = [root_path + "include"]

extra_compile_args = [
    "-std=c++17",
    "-DCMAKE_BUILD_TYPE Release" "-O3",
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    "-D_GLIBCXX_USE_CXX11_ABI=0",
]

if torch.cuda.is_available():
    module = CUDAExtension(
        name="mash_cpp",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
    )
else:
    module = CppExtension(
        name="mash_cpp",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
    )

setup(
    name="MASH-CPP",
    version="0.0.1",
    ext_modules=[module],
    cmdclass={"build_ext": BuildExtension},
)
