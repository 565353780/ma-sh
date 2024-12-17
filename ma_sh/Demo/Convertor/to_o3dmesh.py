from ma_sh.Config.custom_path import toDatasetRootPath
from ma_sh.Module.Convertor.to_o3dmesh import Convertor


def demo():
    dataset_root_folder_path = toDatasetRootPath()
    remove_source = False
    need_normalize = False
    source_data_type = '.ply'
    target_data_type = '.obj'

    if dataset_root_folder_path is None:
        print('[ERROR][to_manifold::demo]')
        print('\t toDatasetRootPath failed!')
        return False

    convertor = Convertor(
        dataset_root_folder_path + "Objaverse_82K/trimesh/",
        dataset_root_folder_path + "Objaverse_82K/o3dmesh/",
        remove_source,
        need_normalize
    )

    convertor.convertAll(source_data_type, target_data_type)
    return True
