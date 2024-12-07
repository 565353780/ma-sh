import os

from ma_sh.Module.Convertor.sample_pcd_objaverse import Convertor


def demo():
    HOME = os.environ["HOME"]

    dataset_root_folder_path = HOME + "/chLi/Dataset/"
    dataset_name = 'Objaverse_82K'
    gt_points_num = 400000
    force_start = False
    worker_num = 16

    convertor = Convertor(
        dataset_root_folder_path,
        dataset_name,
        gt_points_num,
        force_start,
    )

    convertor.convertAll(worker_num)
    return True
