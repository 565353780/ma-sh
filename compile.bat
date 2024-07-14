pip uninstall mash-cpp -y

rmdir ../ma-sh/build
del ../ma-sh/*.egg-info
del ../ma-sh/*.so

python setup.py build_ext --inplace
move compile_commands.json build

pip install .
