import sys
sys.path.append('../ulip-manage/')
sys.path.append('../shapeglot-dataset-manage/')

import os

from ma_sh.Module.Convertor.encode_text import Convertor


def demo():
    HOME = os.environ["HOME"]

    dataset_root_folder_path = HOME + "/chLi/Dataset/"
    device = "cuda:0"
    force_start = True

    convertor = Convertor(
        dataset_root_folder_path,
        device,
        force_start,
    )

    convertor.convertAll()
    return True
