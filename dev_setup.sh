cd ..
git clone git@github.com:565353780/sdf-generate.git
git clone git@github.com:565353780/open3d-manage.git
git clone git@github.com:565353780/param-gauss-recon.git
git clone git@github.com:565353780/open-clip-detect.git
git clone git@github.com:565353780/dino-v2-detect.git
git clone git@github.com:565353780/ulip-manage.git
git clone git@github.com:565353780/distribution-manage.git
git clone git@github.com:565353780/wn-nc.git
git clone git@github.com:565353780/siggraph-rebuttal.git

cd sdf-generate
./dev_setup.sh

cd ../open3d-manage
./dev_setup.sh

cd ../param-gauss-recon
./dev_setup.sh

cd ../open-clip-detect
./dev_setup.sh

cd ../dino-v2-detect
./dev_setup.sh

cd ../ulip-manage
./dev_setup.sh

cd ../distribution-manage
./dev_setup.sh

cd ../wn-nc
./dev_setup.sh

cd ../siggraph-rebuttal
./dev_setup.sh

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
