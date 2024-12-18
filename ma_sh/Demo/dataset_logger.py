from ma_sh.Config.custom_path import toDatasetRootPath
from ma_sh.Module.dataset_logger import DatasetLogger

def demo():
    dataset_root_folder_path = toDatasetRootPath()
    log_folder_path = './logs/'
    target_rel_folder_path_dict = {
        'glbs': 'Objaverse_82K/glbs/',
        'mesh': 'Objaverse_82K/mesh/',
        'manifold': 'Objaverse_82K/manifold/',
        'manifold_pcd': 'Objaverse_82K/manifold_pcd/',
        'mash': 'Objaverse_82K/manifold_mash/',
        'sdf_0_25': 'Objaverse_82K/manifold_sdf_0_25/',
    }
    sleep_second = 60.0

    if dataset_root_folder_path is None:
        print('[ERROR][dataset_logger::demo]')
        print('\t dataset not found!')
        return False

    dataset_logger = DatasetLogger(dataset_root_folder_path, log_folder_path)
    dataset_logger.autoRecordDatasetState(target_rel_folder_path_dict, sleep_second)
    return True
