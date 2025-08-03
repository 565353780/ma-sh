cd ..
git clone https://github.com/565353780/data-convert.git
git clone https://github.com/565353780/sdf-generate.git
git clone https://github.com/565353780/chamfer-distance.git
git clone https://github.com/565353780/mesh-graph-cut.git
git clone https://github.com/565353780/diff-curvature.git
git clone https://github.com/NVlabs/nvdiffrast.git
git clone https://github.com/facebookresearch/pytorch3d.git

# sudo apt install bear cmake libgl1-mesa-glx libgl1-mesa-dev -y

conda install cmake -y

export CC=$(which gcc)
export CXX=$(which g++)
echo "Using CC: $CC"
echo "Using CXX: $CXX"

pip install -U trimesh matplotlib opencv-python einops \
  numpy open3d scipy scikit-learn tqdm numba icecream jax \
  tensorboard

pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124

pip install torch-scatter \
  -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

pip install pyglet==1.5.13

cd sdf-generate
./setup.sh

cd ../chamfer-distance
pip install .

cd ../mesh-graph-cut

COMPILE_MCUT=true
if [ "$(uname)" = "Darwin" ]; then
  if [ -f "./mesh_graph_cut/Lib/mcut/build/bin/libmcut.dylib" ]; then
    COMPILE_MCUT=false
  fi
fi
if [ "$(uname)" = "Linux" ]; then
  if [ -f "./mesh_graph_cut/Lib/mcut/build/bin/libmcut.so" ]; then
    COMPILE_MCUT=false
  fi
fi

if [ $COMPILE_MCUT = true ]; then
  cd ./mesh_graph_cut/Lib/mcut/
  rm -rf build
  mkdir build
  cd build
  cmake ..
  make -j

  cd ../../../../
fi

pip install .

cd ../nvdiffrast
pip install .

cd ../pytorch3d
pip install .

cd ../ma-sh

./compile.sh
