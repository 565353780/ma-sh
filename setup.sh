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

cd sdf-generate
./setup.sh

cd ../open3d-manage
./setup.sh

cd ../param-gauss-recon
./setup.sh

cd ../open-clip-detect
./setup.sh

cd ../dino-v2-detect
./setup.sh

cd ../ulip-manage
./setup.sh

cd ../distribution-manage
./setup.sh

cd ../wn-nc
./setup.sh

cd ../siggraph-rebuttal
./setup.sh

cd ../chamfer-distance
./setup.sh

cd ../ma-sh

if [ "$(uname)" = "Darwin" ]; then
  brew install bear
elif [ "$(uname)" = "Linux" ]; then
  sudo apt install bear -y
  sudo apt install cmake libgl1-mesa-glx libgl1-mesa-dev -y
fi

pip install -U ninja

pip install -U torch torchvision torchaudio

./compile.sh

pip install -U tqdm tensorboard matplotlib trimesh torchviz \
  ftfy regex torch-tb-profiler

pip install git+https://github.com/openai/CLIP.git
