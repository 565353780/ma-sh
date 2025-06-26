if [ "$(uname)" = "Darwin" ]; then
  PROCESSOR_NUM=$(sysctl -n hw.physicalcpu)
elif [ "$(uname)" = "Linux" ]; then
  PROCESSOR_NUM=$(cat /proc/cpuinfo | grep "processor" | wc -l)
fi

export MAX_JOBS=${PROCESSOR_NUM}

export CC=$(which gcc)
export CXX=$(which g++)
echo "Using CC: $CC"
echo "Using CXX: $CXX"

pip uninstall mash-cpp -y

rm -rf ../ma-sh/build
rm -rf ../ma-sh/*.egg-info
rm ../ma-sh/*.so

# bear -- python setup.py build_ext --inplace
python setup.py build_ext --inplace
mv compile_commands.json build

pip install .
