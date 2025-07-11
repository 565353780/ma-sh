import os
import glob
import torch
from platform import system
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDAExtension, CppExtension, BuildExtension

SYSTEM = system()

mash_root_path = os.getcwd() + "/ma_sh/Cpp/"
mash_src_path = mash_root_path + "src/"
mash_sources = glob.glob(mash_src_path + "*.cpp")
mash_include_dirs = [mash_root_path + "include"]

mash_extra_compile_args = [
    "-O3",
    "-DCMAKE_BUILD_TYPE=Release",
    "-D_GLIBCXX_USE_CXX11_ABI=0",
    "-DTORCH_USE_CUDA_DSA",
]

if SYSTEM == "Darwin":
    mash_extra_compile_args.append("-std=c++17")
elif SYSTEM == "Linux":
    mash_extra_compile_args.append("-std=c++17")

if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability()
    arch_str = f"{cc[0]}.{cc[1]}"
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch_str

    mash_sources += glob.glob(mash_src_path + "*.cu")

    extra_compile_args = {
        "cxx": mash_extra_compile_args
        + [
            "-DUSE_CUDA",
            "-DTORCH_USE_CUDA_DSA",
        ],
        "nvcc": [
            "-O3",
            "-Xfatbin",
            "-compress-all",
            "-DUSE_CUDA",
            "-std=c++17",
            "-DTORCH_USE_CUDA_DSA",
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
