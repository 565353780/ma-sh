from ma_sh.Config.custom_path import toDatasetRootPath
from ma_sh.Module.Convertor.sample_pcd import Convertor


if __name__ == '__main__':
    dataset_root_folder_path = toDatasetRootPath()
    gt_points_num = 4000000000
    # gt_points_num = 1024
    random_weight = -1.0
    noise_weight = -1.0
    need_normalize = False
    source_data_type = '.bin'
    target_data_type = '.ply'

    if dataset_root_folder_path is None:
        print('[ERROR][sample_pcd::demo]')
        print('\t toDatasetRootPath failed!')
        exit()

    convertor = Convertor(
        dataset_root_folder_path + "KITTI/raw_data/city/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/",
        dataset_root_folder_path + "KITTI/pcd/",
        gt_points_num,
        random_weight,
        noise_weight,
        need_normalize,
    )

    convertor.convertAll(source_data_type, target_data_type)
