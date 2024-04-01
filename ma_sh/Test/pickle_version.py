import numpy as np
from ma_sh.Model.mash import Mash


def test():
    mash_params_file_path = "/home/chli/Dataset/aro_net/data/shapenet/mash/02691156/92b7d0035cefb816d13ef00338ba8c52_obj.npy"

    mash = Mash.fromParamsFile(mash_params_file_path)
    print(mash.mask_params[0])

    data_dict = np.load(mash_params_file_path, allow_pickle=True).item()
    print(data_dict["mask_params"][0])
    return True
