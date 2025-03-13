from ma_sh.Config.custom_path import toDatasetRootPath
from ma_sh.Module.Convertor.sample_pcd import Convertor


if __name__ == '__main__':
    dataset_root_folder_path = toDatasetRootPath()
    gt_points_num = 400000
    gt_points_num = 4096
    random_weight = 10.0
    noise_weight = 0.002
    need_normalize = False
    source_data_type = '.obj'
    target_data_type = '.ply'

    if dataset_root_folder_path is None:
        print('[ERROR][sample_pcd::demo]')
        print('\t toDatasetRootPath failed!')
        exit()

    convertor = Convertor(
        dataset_root_folder_path + "ShapeNet/manifold/03001627/",
        dataset_root_folder_path + "ShapeNet/manifold_pcd-4096_random-10_noise-0-002/03001627/",
        gt_points_num,
        random_weight,
        noise_weight,
        need_normalize,
    )

    convertor.convertAll(source_data_type, target_data_type)
