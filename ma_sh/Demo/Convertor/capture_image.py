import sys

sys.path.append("../open3d-manage")

import os

from ma_sh.Module.Convertor.capture_image import Convertor


def demo():
    HOME = os.environ["HOME"]

    dataset_root_folder_path = HOME + "/chLi/Dataset/"
    window_name = "Shape Image Sampler"
    width = 224
    height = 224
    left = 10
    top = 10
    visible = False
    y_rotate_num = 8
    x_rotate_num = 5
    force_start = False

    convertor = Convertor(
        dataset_root_folder_path,
        window_name,
        width,
        height,
        left,
        top,
        visible,
        y_rotate_num,
        x_rotate_num,
        force_start,
    )

    convertor.convertAll()
    return True
