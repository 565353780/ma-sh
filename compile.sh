TORCH_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')
PYBIND11_PATH=$(python -c 'import pybind11;print(pybind11.__path__[0])')

rm -rf build
mkdir build
cd build

cmake \
	-DCMAKE_PREFIX_PATH=$TORCH_PATH \
	-Dpybind11_DIR=$PYBIND11_PATH \
	..
