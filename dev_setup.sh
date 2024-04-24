cd ..
git clone git@github.com:565353780/sdf-manage.git
git clone git@github.com:565353780/open3d-manage.git

cd sdf-manage
./dev_setup.sh

cd ../open3d-manage
./dev_setup.sh

if [ "$(uname)" == "Darwin" ]; then
	brew install bear
elif [ "$(uname)" = "Linux" ]; then
	sudo apt install bear -y
fi

pip install -U ninja

pip install -U torch torchvision torchaudio

cd ../ma-sh
./compile.sh

pip install -U tqdm tensorboard matplotlib trimesh torchviz \
	ftfy regex

pip install git+https://github.com/openai/CLIP.git
