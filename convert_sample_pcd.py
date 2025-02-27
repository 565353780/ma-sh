from ma_sh.Config.custom_path import toDatasetRootPath
from ma_sh.Module.Convertor.sample_pcd import Convertor


if __name__ == '__main__':
    dataset_root_folder_path = toDatasetRootPath()
    gt_points_num = 400000
    source_data_type = '.obj'
    target_data_type = '.npy'

    if dataset_root_folder_path is None:
        print('[ERROR][sample_pcd::demo]')
        print('\t toDatasetRootPath failed!')
        exit()

    convertor = Convertor(
        dataset_root_folder_path + "vae-eval/manifold/",
        dataset_root_folder_path + "vae-eval/manifold_pcd/",
        gt_points_num,
    )

    convertor.convertAll(source_data_type, target_data_type)
