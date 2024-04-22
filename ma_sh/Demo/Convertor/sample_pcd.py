import os

from ma_sh.Module.Convertor.sample_pcd import Convertor


def demo():
    HOME = os.environ["HOME"]

    shape_root_folder_path = HOME + "/chLi/Dataset/ShapeNet/Core/ShapeNetCore.v2/"
    shape_root_folder_path = HOME + "/chLi/Dataset/SDF/ShapeNet/manifold/"
    save_root_folder_path = HOME + "/chLi/Dataset/Mash/ShapeNet/"

    force_start = True
    gt_points_num = 400000

    convertor = Convertor(
        shape_root_folder_path,
        save_root_folder_path,
        force_start,
        gt_points_num,
    )

    convertor.convertAll()
    return True
