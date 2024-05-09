import os

from ma_sh.Module.Convertor.mash_pcd import Convertor


def demo():
    HOME = os.environ["HOME"]

    dataset_root_folder_path = HOME + "/chLi/Dataset/"
    device = "cpu"
    force_start = False

    convertor = Convertor(
        dataset_root_folder_path,
        device,
        force_start,
    )

    convertor.convertAll()
    return True
