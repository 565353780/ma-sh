from ma_sh.Test.ones_like import test as test_ones_like
from ma_sh.Test.grad import test as test_grad
from ma_sh.Test.mash_unit import test as test_mash_unit
from ma_sh.Test.mash_model import test as test_mash_model

if __name__ == "__main__":
    test_ones_like()
    test_grad()
    test_mash_unit()
    test_mash_model()
