from ma_sh.Test.import_o3d import test as test_import_o3d
from ma_sh.Test.rotate import test as test_rotate
from ma_sh.Test.grad import test as test_grad
from ma_sh.Test.mash_unit import test as test_mash_unit
from ma_sh.Test.mash_model import test as test_mash_model
from ma_sh.Test.data_check import test as test_check_data
from ma_sh.Test.metric import test as test_metric
from ma_sh.Test.simple_mash_model import test as test_simple_mash_model
from ma_sh.Test.simple_mash_render import test as test_render_simple_mash
from ma_sh.Test.random_mash import test as test_random_mash
from ma_sh.Test.param_coverage import test as test_param_coverage

if __name__ == "__main__":
    # test_metric()
    # test_check_data()
    # test_import_o3d()
    # test_rotate()
    # test_grad()
    # test_mash_unit()
    # test_mash_model()
    # test_simple_mash_model()
    # test_render_simple_mash()
    # test_random_mash()
    test_param_coverage()
