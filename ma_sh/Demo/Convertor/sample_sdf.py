import sys
sys.path.append("../sdf-generate")

from ma_sh.Config.custom_path import toDatasetRootPath
from ma_sh.Module.Convertor.sample_sdf import Convertor


def demo(gauss_noise: float = 0.0025):
    dataset_root_folder_path = toDatasetRootPath()
    if dataset_root_folder_path is None:
        print('[ERROR][sample_sdf::demo]')
        print('\t toDatasetRootPath failed!')
        return False

    sample_sdf_point_num = 250000
    # gauss_noise = 0.0025
    source_data_type = '.obj'
    target_data_type = '.npy'

    noise_label = str(gauss_noise).replace('.', '_')

    convertor = Convertor(
        dataset_root_folder_path + '/Objaverse_82K/manifold/',
        dataset_root_folder_path + '/Objaverse_82K/sdf_' + noise_label + '/',
        sample_sdf_point_num,
        gauss_noise,
    )

    convertor.convertAll(source_data_type, target_data_type)
    return True
