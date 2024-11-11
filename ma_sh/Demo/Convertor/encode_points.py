import sys
sys.path.append('../ulip-manage/')

import os

from ma_sh.Module.Convertor.encode_points import Convertor


def demo():
    HOME = os.environ["HOME"]

    dataset_root_folder_path = HOME + "/chLi/Dataset/"
    sample_point_num_list = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    device = "cuda:0"
    force_start = False

    convertor = Convertor(
        dataset_root_folder_path,
        sample_point_num_list,
        device,
        force_start,
    )

    convertor.convertAll()
    return True
