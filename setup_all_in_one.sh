cd ..
git clone https://github.com/565353780/sdf-generate.git
git clone https://github.com/565353780/open3d-manage.git
git clone https://github.com/565353780/param-gauss-recon.git
git clone https://github.com/565353780/open-clip-detect.git
git clone https://github.com/565353780/dino-v2-detect.git
git clone https://github.com/565353780/ulip-manage.git
git clone https://github.com/565353780/distribution-manage.git
git clone https://github.com/565353780/wn-nc.git
git clone https://github.com/565353780/siggraph-rebuttal.git
git clone https://github.com/565353780/chamfer-distance.git
git clone https://github.com/565353780/mesh-graph-cut.git
git clone https://github.com/565353780/diff-curvature.git
git clone https://github.com/NVlabs/nvdiffrast.git
git clone https://github.com/facebookresearch/pytorch3d.git

# sudo apt install libc++-dev libc++abi-dev libcgal-dev libomp-dev bear \
#   cmake libgl1-mesa-glx libgl1-mesa-dev -y

conda install cmake -y

export CC=$(which gcc)
export CXX=$(which g++)
echo "Using CC: $CC"
echo "Using CXX: $CXX"

pip install -U tqdm trimesh scikit-image open3d numpy gradio plotly \
  opencv-python icecream jax ninja tensorboard matplotlib torchviz ftfy \
  regex torch-tb-profiler scipy scikit-learn numba einops

pip install pyvista==0.44.1

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.1+cu128.html

pip install pyglet==1.5.13

cd wn-nc/wn_nc/Lib/ANN/

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
pip install .

cd ../../../chamfer-distance
pip install .

cd ../mesh-graph-cut
pip install .

cd ../nvdiffrast
pip install .

cd ../pytorch3d
pip install .

cd ../ma-sh

./compile.sh
