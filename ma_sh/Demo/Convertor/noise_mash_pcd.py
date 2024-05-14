import os

from ma_sh.Module.Convertor.noise_mash_pcd import Convertor


def demo(gauss_sigma: float = 0.01):
    HOME = os.environ["HOME"]

    dataset_root_folder_path = HOME + "/chLi/Dataset/"
    #gauss_sigma = 0.01
    device = "cpu"
    force_start = False

    convertor = Convertor(
        dataset_root_folder_path,
        gauss_sigma,
        device,
        force_start,
    )

    convertor.convertAll()
    return True
