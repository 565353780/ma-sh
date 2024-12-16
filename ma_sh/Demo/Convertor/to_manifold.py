import sys

sys.path.append("../sdf-generate")

from ma_sh.Config.custom_path import toDatasetRootPath
from ma_sh.Module.Convertor.to_manifold import Convertor


def demo():
    dataset_root_folder_path = toDatasetRootPath()
    if dataset_root_folder_path is None:
        print('[ERROR][to_manifold::demo]')
        print('\t toDatasetRootPath failed!')
        return False

    force_start = False

    convertor = Convertor(
        dataset_root_folder_path,
        force_start,
    )

    convertor.convertAll()
    return True
