cd ..
git clone https://github.com/facebookresearch/pytorch3d.git
git clone https://github.com/pybind/pybind11.git ./ma-sh/ma_sh/Cpp/pybind11

pip install torch torchvision torchaudio

cd pytorch3d
rm -rf build
pip install -e .

pip install tqdm
