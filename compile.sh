export MAX_JOBS=32

pip uninstall mash-cpp -y

# rm -rf ./build
rm ./build/compile_commands.json

rm -rf ./*.egg-info
rm ./*.so

bear -- python setup.py build_ext --inplace
mv compile_commands.json build

pip install .
