from ma_sh.Config.custom_path import toDatasetRootPath
from ma_sh.Module.Convertor.sample_pcd import Convertor


if __name__ == "__main__":
    dataset_root_folder_path = toDatasetRootPath()
    gt_points_num = 4000000000
    gt_points_num = 8192
    random_weight = 10.0
    random_weight = 0.0
    noise_weight = 0.0
    need_normalize = False
    source_data_type = ".obj"
    target_data_type = ".ply"

    if dataset_root_folder_path is None:
        print("[ERROR][sample_pcd::demo]")
        print("\t toDatasetRootPath failed!")
        exit()

    convertor = Convertor(
        dataset_root_folder_path + "Thingi10K/mesh/",
        dataset_root_folder_path + "Thingi10K/mesh_pcd_8192/",
        gt_points_num,
        random_weight,
        noise_weight,
        need_normalize,
    )

    convertor.convertAll(source_data_type, target_data_type)
