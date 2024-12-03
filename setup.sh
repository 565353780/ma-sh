cd ..
git clone https://github.com/565353780/sdf-generate.git
git clone https://github.com/565353780/open3d-manage.git
git clone https://github.com/565353780/param-gauss-recon.git

cd sdf-generate
./setup.sh

cd ../open3d-manage
./setup.sh

cd ../param-gauss-recon
./setup.sh

if [[ "$(uname)" = "Darwin" ]]; then
  brew install bear
elif [[ "$(uname)" = "Linux" ]]; then
  sudo apt install bear -y
fi

pip install -U ninja

pip install -U torch torchvision torchaudio

cd ../ma-sh
./compile.sh

pip install -U tqdm tensorboard matplotlib trimesh torchviz \
  ftfy regex

pip install git+https://github.com/openai/CLIP.git
