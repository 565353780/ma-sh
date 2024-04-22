import os

from ma_sh.Module.Convertor.normalize_mesh import Convertor


def demo():
    HOME = os.environ["HOME"]

    shape_root_folder_path = HOME + "/chLi/Dataset/ShapeNet/Core/ShapeNetCore.v2/"
    shape_root_folder_path = HOME + "/chLi/Dataset/SDF/ShapeNet/manifold/"
    save_root_folder_path = HOME + "/chLi/Dataset/Mash/ShapeNet/"

    force_start = True

    convertor = Convertor(
        shape_root_folder_path,
        save_root_folder_path,
        force_start,
    )

    convertor.convertAll()
    return True
