UNAME=$(uname -s)

if [ "$UNAME" == "Darwin" ]; then
	brew install bear
elif [ "$UNAME" == "Linux" ]; then
	sudo apt install bear -y
fi

pip install -U ninja

pip install -U torch torchvision torchaudio

cd ../ma-sh
./compile.sh

pip install -U tqdm tensorboard matplotlib trimesh torchviz
pip install open3d==0.15.1
