cd ..
git clone https://github.com/facebookresearch/pytorch3d.git

pip install torch torchvision torchaudio

cd pytorch3d
rm -rf build
pip install -e .

pip install tqdm
