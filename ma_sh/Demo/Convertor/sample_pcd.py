from ma_sh.Config.custom_path import toDatasetRootPath
from ma_sh.Module.Convertor.sample_pcd import Convertor


def demo():
    dataset_root_folder_path = toDatasetRootPath()
    gt_points_num = 400000
    random_weight = -1.0
    noise_weight = -1.0
    need_normalize = False
    source_data_type = '.obj'
    target_data_type = '.ply'

    if dataset_root_folder_path is None:
        print('[ERROR][sample_pcd_objaverse::demo]')
        print('\t toDatasetRootPath failed!')
        return False

    convertor = Convertor(
        dataset_root_folder_path + "Objaverse_82K/manifold/",
        dataset_root_folder_path + "Objaverse_82K/manifold_pcd/",
        gt_points_num,
        random_weight,
        noise_weight,
        need_normalize,
    )

    convertor.convertAll(source_data_type, target_data_type)
    return True
