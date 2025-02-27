from ma_sh.Config.custom_path import toDatasetRootPath
from ma_sh.Module.Convertor.mash_to_mesh import Convertor


def demo():
    dataset_root_folder_path = toDatasetRootPath()
    source_data_type = '.npy'
    target_data_type = '.ply'

    if dataset_root_folder_path is None:
        print('[ERROR][mash_to_mesh::demo]')
        print('\t toDatasetRootPath failed!')
        return False

    convertor = Convertor(
        dataset_root_folder_path + "vae-eval/manifold_mash_400/",
        dataset_root_folder_path + "vae-eval/manifold_recon_mesh_400/",
    )

    convertor.convertAll(source_data_type, target_data_type)
    return True
