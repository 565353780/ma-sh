from ma_sh.Config.custom_path import toDatasetRootPath
from ma_sh.Module.Convertor.mash_to_mesh import Convertor


def demo(
    mash_rel_folder_path: str,
    save_mesh_rel_folder_path: str,
):
    dataset_root_folder_path = toDatasetRootPath()
    source_data_type = '.npy'
    target_data_type = '.ply'

    if dataset_root_folder_path is None:
        print('[ERROR][mash_to_mesh::demo]')
        print('\t toDatasetRootPath failed!')
        return False

    convertor = Convertor(
        dataset_root_folder_path + mash_rel_folder_path,
        dataset_root_folder_path + save_mesh_rel_folder_path,
    )

    convertor.convertAll(source_data_type, target_data_type)
    return True
