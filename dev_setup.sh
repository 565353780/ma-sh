sudo apt install bear -y

pip install -U ninja

pip install -U torch torchvision torchaudio

cd ../ma-sh
./compile.sh

pip install -U tqdm open3d tensorboard matplotlib trimesh torchviz
