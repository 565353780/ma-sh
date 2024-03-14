from ma_sh.Test.ones_like import test as test_ones_like
from ma_sh.Test.grad import test as test_grad
from ma_sh.Test.mash_unit import test as test_mash_unit
from ma_sh.Test.speed import test as test_speed
from ma_sh.Test.mash_kernel import test as test_mash_kernel
from ma_sh.Test.mash_model import test as test_mash_model

if __name__ == "__main__":
    test_ones_like()
    test_grad()
    test_mash_unit()
    test_speed()
    test_mash_kernel()
    test_mash_model()
