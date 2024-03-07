export MAX_JOBS=32

pip uninstall mash-cpp -y

rm -rf ./build
rm -rf ./*.egg-info
rm ./*.so

mkdir build
cd build
cmake \
	-DCMAKE_PREFIX_PATH=$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)') \
	..

make -j

cp ./mash_cpp*.so ../
