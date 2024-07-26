cd ..
git clone https://github.com/565353780/param-gauss-recon.git

cd param-gauss-recon
setup.bat

pip install -U ninja

pip install -U torch torchvision torchaudio

cd ../ma-sh
compile.bat

pip install -U tqdm tensorboard matplotlib trimesh torchviz ftfy regex open3d

pip install git+https://github.com/openai/CLIP.git
