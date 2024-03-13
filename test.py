from ma_sh.Test.ones_like import test as test_ones_like
from ma_sh.Test.grad import test as test_grad
from ma_sh.Test.sh_points import test as test_sh_points
from ma_sh.Test.speed import test as test_speed
from ma_sh.Test.mash import test as test_mash

if __name__ == "__main__":
    test_ones_like()
    test_grad()
    test_sh_points()
    test_speed()
    test_mash()
