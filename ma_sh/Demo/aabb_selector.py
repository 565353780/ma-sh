import torch

from ma_sh.Model.mash import Mash
from ma_sh.Module.aabb_selector  import AABBSelector

def demo():
    mash_file_path = '/home/chli/Dataset/MashV4/ShapeNet/03001627/1006be65e7bc937e9141f9b58470d646.npy'
    mash_file_path = '/home/chli/Dataset/MashV4/ShapeNet/03001627/1007e20d5e811b308351982a6e40cf41.npy'

    mash = Mash.fromParamsFile(
        mash_file_path,
        40,
        100,
        1.0,
        torch.int64,
        torch.float32,
        'cuda',
    )

    aabb_selector = AABBSelector(mash)

    aabb_selector.run()
    return True
