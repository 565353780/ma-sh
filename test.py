from ma_sh.Test.cpp import test as test_cpp
from ma_sh.Test.ones_like import test as test_ones_like
from ma_sh.Test.torch_multi import test as test_torch_multi

if __name__ == "__main__":
    test_cpp()
    test_ones_like()
    test_torch_multi()
