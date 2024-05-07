import os

from ma_sh.Module.Convertor.encode_mash import Convertor


def demo():
    HOME = os.environ["HOME"]

    dataset_root_folder_path = HOME + "/chLi/Dataset/"
    model_file_path = "../mash-autoencoder/output/pretrain-10dim-v3/model_best.pth"
    device = "cuda:0"
    force_start = False

    convertor = Convertor(
        dataset_root_folder_path,
        model_file_path,
        device,
        force_start,
    )

    convertor.convertAll()
    return True
