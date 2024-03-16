export MAX_JOBS=32

pip uninstall pointnet2-ops -y

rm -rf ../ma-sh/ma_sh/Lib/pointnet2_ops/build
rm -rf ../ma-sh/ma_sh/Lib/pointnet2_ops/*.egg-info
rm ../ma-sh/ma_sh/Lib/pointnet2_ops/*.so

pip install ../ma-sh/ma_sh/Lib/pointnet2_ops

pip uninstall mash-cpp -y

rm -rf ../ma-sh/build
rm -rf ../ma-sh/*.egg-info
rm ../ma-sh/*.so

bear -- python setup.py build_ext --inplace
mv compile_commands.json build
