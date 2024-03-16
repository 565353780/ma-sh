import os
import glob
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

pointnet2_ops_root_path = os.getcwd() + "/"
pointnet2_ops_src_path = pointnet2_ops_root_path + "src/"
pointnet2_ops_sources = glob.glob(pointnet2_ops_src_path + "*.cpp") + glob.glob(
    pointnet2_ops_src_path + "*.cu"
)
pointnet2_ops_include_dirs = [pointnet2_ops_root_path + "include"]

os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6"

extra_compile_args = {
    "cxx": [
        "-O3",
        "-std=c++17",
        "-DCMAKE_BUILD_TYPE Release",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    ],
    "nvcc": [
        "-O3",
        "-Xfatbin",
        "-compress-all",
    ],
}

pointnet2_ops_module = CUDAExtension(
    name="pointnet2_ops",
    sources=pointnet2_ops_sources,
    include_dirs=pointnet2_ops_include_dirs,
    extra_compile_args=extra_compile_args,
)


setup(
    name="pointnet2_ops",
    version="3.0.0",
    author="Erik Wijmans",
    packages=find_packages(),
    ext_modules=[pointnet2_ops_module],
    cmdclass={"build_ext": BuildExtension},
    include_package_data=True,
)
