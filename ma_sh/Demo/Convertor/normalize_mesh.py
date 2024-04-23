import os

from ma_sh.Module.Convertor.normalize_mesh import Convertor


def demo():
    HOME = os.environ["HOME"]

    shape_root_folder_path = HOME + "/chLi/Dataset/ShapeNet/Core/ShapeNetCore.v2/"
    dataset_root_folder_path = HOME + "/chLi/Dataset/"

    force_start = False

    convertor = Convertor(
        shape_root_folder_path,
        dataset_root_folder_path,
        force_start,
    )

    convertor.convertAll()
    return True
