# sudo apt install libc++-dev libc++abi-dev libcgal-dev libomp-dev bear \
#   cmake libgl1-mesa-glx libgl1-mesa-dev -y

conda install cmake

# export CC=$(which gcc)
# export CXX=$(which g++)
# echo "Using CC: $CC"
# echo "Using CXX: $CXX"

pip install -U tqdm trimesh scikit-image open3d numpy gradio plotly \
  opencv-python icecream jax ninja tensorboard matplotlib torchviz ftfy \
  regex torch-tb-profiler

pip install pyvista==0.44.1

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124

cd ../wn-nc/wn_nc/Lib/ANN/

rm -rf build

mkdir build
cd build

cmake .. -DCMAKE_INSTALL_PREFIX=./install
make -j
make install

cd ../../../..
rm -rf bin

./wn_nc/Bash/build_GR_cpu.sh
./wn_nc/Bash/build_GR_cuda.sh

cd wn_nc/Cpp
pip install -e .

cd ../../../chamfer-distance
pip install -e .

cd ../mesh-graph-cut
pip install -e .

cd ../ma-sh

./compile.sh
