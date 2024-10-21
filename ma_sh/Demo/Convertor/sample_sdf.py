import sys

sys.path.append("../sdf-generate")
import os

from ma_sh.Module.Convertor.sample_sdf import Convertor


def demo(gauss_noise: float = 0.0025):
    HOME = os.environ["HOME"]

    dataset_root_folder_path = HOME + "/chLi/Dataset/"
    sample_sdf_point_num = 250000
    # gauss_noise = 0.0025
    force_start = True

    convertor = Convertor(
        dataset_root_folder_path,
        sample_sdf_point_num,
        gauss_noise,
        force_start,
    )

    convertor.convertAll()
    return True
