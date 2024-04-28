from ma_sh.Test.import_o3d import test as test_import_o3d
from ma_sh.Test.rotate import test as test_rotate
from ma_sh.Test.grad import test as test_grad
from ma_sh.Test.mash_unit import test as test_mash_unit
from ma_sh.Test.mash_model import test as test_mash_model
from ma_sh.Test.data_check import test as test_check_data

if __name__ == "__main__":
    test_check_data()
    exit()
    test_import_o3d()
    test_rotate()
    test_grad()
    test_mash_unit()
    test_mash_model()
