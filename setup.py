import os
import glob
import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDAExtension, CppExtension, BuildExtension


mash_root_path = os.getcwd() + "/../ma-sh/ma_sh/Cpp/"
mash_src_path = mash_root_path + "src/"
mash_sources = glob.glob(mash_src_path + "*.cpp")
mash_include_dirs = [mash_root_path + "include"]

mash_extra_compile_args = [
    "-O3",
    "-std=c++17",
    "-DCMAKE_BUILD_TYPE Release",
    "-D_GLIBCXX_USE_CXX11_ABI=0",
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
]

if torch.cuda.is_available():
    os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6;8.9"

    mash_sources += glob.glob(mash_src_path + "*.cu")

    extra_compile_args = {
        "cxx": mash_extra_compile_args + ["-DUSE_CUDA"],
        "nvcc": [
            "-O3",
            "-Xfatbin",
            "-compress-all",
            "-DUSE_CUDA",
        ],
    }

    mash_module = CUDAExtension(
        name="mash_cpp",
        sources=mash_sources,
        include_dirs=mash_include_dirs,
        extra_compile_args=extra_compile_args,
    )

else:
    mash_module = CppExtension(
        name="mash_cpp",
        sources=mash_sources,
        include_dirs=mash_include_dirs,
        extra_compile_args=mash_extra_compile_args,
    )

setup(
    name="MASH-CPP",
    version="1.0.0",
    author="Changhao Li",
    packages=find_packages(),
    ext_modules=[mash_module],
    cmdclass={"build_ext": BuildExtension},
    include_package_data=True,
)
