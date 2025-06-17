cd ..
git clone git@github.com:565353780/chamfer-distance.git
git clone git@github.com:565353780/mesh-graph-cut.git

cd chamfer-distance
./dev_setup.sh

cd ../mesh-graph-cut
./dev_setup.sh

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
