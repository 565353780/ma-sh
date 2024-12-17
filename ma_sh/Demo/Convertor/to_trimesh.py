from ma_sh.Config.custom_path import toDatasetRootPath
from ma_sh.Module.Convertor.to_trimesh import Convertor


def demo():
    dataset_root_folder_path = toDatasetRootPath()
    remove_source = False
    need_normalize = True
    source_data_type = '.glb'
    target_data_type = '.ply'

    if dataset_root_folder_path is None:
        print('[ERROR][to_manifold::demo]')
        print('\t toDatasetRootPath failed!')
        return False

    convertor = Convertor(
        dataset_root_folder_path + "Objaverse_82K/glbs/",
        dataset_root_folder_path + "Objaverse_82K/trimesh/",
        remove_source,
        need_normalize
    )

    convertor.convertAll(source_data_type, target_data_type)
    return True
