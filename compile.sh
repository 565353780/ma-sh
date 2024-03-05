TORCH_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')

rm -rf build
mkdir build
cd build

cmake \
	-DCMAKE_PREFIX_PATH=$TORCH_PATH \
	..

make -j
