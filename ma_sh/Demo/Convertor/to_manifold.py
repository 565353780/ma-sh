import sys

sys.path.append("../sdf-generate")
import os

from ma_sh.Module.Convertor.to_manifold import Convertor


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
