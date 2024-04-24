import os

from ma_sh.Module.Convertor.encode_image import Convertor


def demo():
    HOME = os.environ["HOME"]

    dataset_root_folder_path = HOME + "/chLi/Dataset/"
    clip_model_id = "ViT-L/14"
    device = "cpu"
    force_start = False

    convertor = Convertor(
        dataset_root_folder_path,
        clip_model_id,
        device,
        force_start,
    )

    convertor.convertAll()
    return True
