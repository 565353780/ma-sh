pip install -U ninja

pip install -U torch torchvision torchaudio

cd ../ma-sh
./compile.sh

pip install -U tqdm tensorboard matplotlib trimesh torchviz \
	ftfy regex

pip install git+https://github.com/openai/CLIP.git
