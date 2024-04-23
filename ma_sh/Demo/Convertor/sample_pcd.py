import os

from ma_sh.Module.Convertor.sample_pcd import Convertor


def demo():
    HOME = os.environ["HOME"]

    dataset_root_folder_path = HOME + "/chLi/Dataset/"
    gt_points_num = 400000
    force_start = False

    convertor = Convertor(
        dataset_root_folder_path,
        gt_points_num,
        force_start,
    )

    convertor.convertAll()
    return True
