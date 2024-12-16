import sys

sys.path.append("../sdf-generate")

from ma_sh.Config.custom_path import toDatasetRootPath
from ma_sh.Module.Convertor.to_manifold import Convertor


def demo():
    dataset_root_folder_path = toDatasetRootPath()
    source_data_type = '.obj'
    target_data_type = '.obj'

    if dataset_root_folder_path is None:
        print('[ERROR][to_manifold::demo]')
        print('\t toDatasetRootPath failed!')
        return False

    convertor = Convertor(
        dataset_root_folder_path + "Objaverse_82K/mesh/",
        dataset_root_folder_path + "Objaverse_82K/manifold/",
    )

    convertor.convertAll(source_data_type, target_data_type)
    return True
