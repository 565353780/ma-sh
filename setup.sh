cd ..
git clone https://github.com/565353780/sdf-generate.git
git clone https://github.com/565353780/open3d-manage.git
git clone https://github.com/565353780/param-gauss-recon.git
git clone https://github.com/565353780/open-clip-detect.git
git clone https://github.com/565353780/dino-v2-detect.git
git clone https://github.com/565353780/ulip-manage.git

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

if [ "$(uname)" = "Darwin" ]; then
  brew install bear
elif [ "$(uname)" = "Linux" ]; then
  sudo apt install bear -y
  sudo apt install cmake libgl1-mesa-glx libgl1-mesa-dev -y
fi

pip install -U ninja

pip install -U torch torchvision torchaudio

cd ../ma-sh
./compile.sh

pip install -U tqdm tensorboard matplotlib trimesh torchviz \
  ftfy regex torch-tb-profiler

pip install git+https://github.com/openai/CLIP.git
