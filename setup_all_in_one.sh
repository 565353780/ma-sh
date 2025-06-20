cd ..
git clone https://github.com:565353780/data-convert.git
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
  numpy open3d scipy scikit-learn tqdm numba icecream jax

pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124

pip install torch-scatter \
  -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

pip install pyglet==1.5.13

cd chamfer-distance
pip install .

cd ../mesh-graph-cut
pip install .

cd ../nvdiffrast
pip install .

cd ../pytorch3d
pip install .

cd ../ma-sh

./compile.sh
