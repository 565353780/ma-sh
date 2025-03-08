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

    data_tag = '_1600_sh3'
    data_tag = '_400'
    data_tag = '_random'
    data_tag = ''

    convertor = Convertor(
        dataset_root_folder_path + "ShapeNet/manifold_mash" + data_tag + "/03001627/",
        dataset_root_folder_path + "ShapeNet/manifold_recon_mesh" + data_tag + "/03001627/",
    )

    convertor.convertAll(source_data_type, target_data_type)
    return True
