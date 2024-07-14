pip uninstall mash-cpp -y

rmdir ../ma-sh/build
rmdir ../ma-sh/*.egg-info
rm ../ma-sh/*.so

python setup.py build_ext --inplace
mv compile_commands.json build

pip install .
