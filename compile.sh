export MAX_JOBS=32

pip uninstall pointnet2-ops mash-cpp -y

rm -rf ./build
rm -rf ./*.egg-info
rm ./*.so

bear -- python setup.py build_ext --inplace
mv compile_commands.json build

pip install .
