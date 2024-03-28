export MAX_JOBS=32

# pip uninstall mash-cpp -y

# rm -rf ../ma-sh/build
# rm -rf ../ma-sh/*.egg-info
# rm ../ma-sh/*.so

bear -- python setup.py build_ext --inplace
mv compile_commands.json build

pip install .
