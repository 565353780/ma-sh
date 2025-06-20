cd ..
git clone https://github.com/565353780/data-convert.git
git clone https://github.com/565353780/sdf-generate.git
git clone https://github.com/565353780/chamfer-distance.git
git clone https://github.com/565353780/mesh-graph-cut.git

cd data-convert
./setup.sh

cd ../sdf-generate
./setup.sh

cd ../chamfer-distance
./setup.sh

cd ../mesh-graph-cut
./setup.sh

cd ../ma-sh

if [ "$(uname)" = "Darwin" ]; then
  brew install bear
elif [ "$(uname)" = "Linux" ]; then
  sudo apt install bear cmake libgl1-mesa-glx libgl1-mesa-dev -y
fi

pip install -U ninja tensorboard trimesh

pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124

./compile.sh
