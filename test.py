from ma_sh.Test.ones_like import test as test_ones_like
from ma_sh.Test.grad import test as test_grad
from ma_sh.Test.mash_unit import test as test_mash_unit
from ma_sh.Test.mash_model import test as test_mash_model
from ma_sh.Test.cpp_with_cuda import test as test_cpp_with_cuda

if __name__ == "__main__":
    test_cpp_with_cuda()
    test_ones_like()
    test_grad()
    test_mash_unit()
    test_mash_model()
