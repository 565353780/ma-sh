import os

from ma_sh.Module.Convertor.sample_noise_pcd import Convertor


def demo():
    HOME = os.environ["HOME"]

    dataset_root_folder_path = HOME + "/chLi/Dataset/"
    gt_points_num = 400000
    gauss_sigma = 0.05
    force_start = False

    convertor = Convertor(
        dataset_root_folder_path,
        gt_points_num,
        gauss_sigma,
        force_start,
    )

    convertor.convertAll()
    return True
