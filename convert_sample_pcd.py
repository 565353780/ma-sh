from ma_sh.Config.custom_path import toDatasetRootPath
from ma_sh.Module.Convertor.sample_pcd import Convertor


if __name__ == '__main__':
    dataset_root_folder_path = toDatasetRootPath()
    gt_points_num = 400000
    random_weight = -1.0
    source_data_type = '.obj'
    target_data_type = '.ply'

    if dataset_root_folder_path is None:
        print('[ERROR][sample_pcd::demo]')
        print('\t toDatasetRootPath failed!')
        exit()

    gt_points_num = 4096

    convertor = Convertor(
        dataset_root_folder_path + "ShapeNet/manifold/03001627/",
        dataset_root_folder_path + "ShapeNet/manifold_pcd_fps/" + str(gt_points_num) + "/03001627/",
        gt_points_num,
        random_weight,
    )

    convertor.convertAll(source_data_type, target_data_type)
