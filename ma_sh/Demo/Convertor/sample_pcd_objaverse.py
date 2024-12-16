from ma_sh.Config.custom_path import toDatasetRootPath
from ma_sh.Module.Convertor.sample_pcd_objaverse import Convertor


def demo():
    dataset_root_folder_path = toDatasetRootPath()
    if dataset_root_folder_path is None:
        print('[ERROR][sample_pcd_objaverse::demo]')
        print('\t toDatasetRootPath failed!')
        return False

    dataset_name = 'Objaverse_82K'
    gt_points_num = 400000
    worker_num = 1

    convertor = Convertor(
        dataset_root_folder_path,
        dataset_name,
        gt_points_num,
    )

    convertor.convertAll(worker_num)
    return True
