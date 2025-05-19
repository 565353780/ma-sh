import sys

sys.path.append("../distribution-manage/")

from ma_sh.Config.custom_path import toDatasetRootPath
from ma_sh.Method.mash_distribution import getMashDistribution

if __name__ == "__main__":
    dataset_root_folder_path = toDatasetRootPath()

    if dataset_root_folder_path is None:
        print("[ERROR][get_mash_distribution::__main__]")
        print("\t dataset not found!")
        exit()

    mash_folder_path = dataset_root_folder_path + "MashV4/"
    save_transformers_folder_path = dataset_root_folder_path + "Transformers/"
    overwrite = False

    print("dataset_root_folder_path:", dataset_root_folder_path)
    getMashDistribution(
        dataset_root_folder_path + "MashV4/",
        "ShapeNet",
        save_transformers_folder_path,
        overwrite,
    )
    getMashDistribution(
        dataset_root_folder_path + "MashV4/ShapeNet/03001627/",
        "ShapeNet_03001627",
        save_transformers_folder_path,
        overwrite,
    )
    getMashDistribution(
        dataset_root_folder_path + "Objaverse_82K/manifold_mash/",
        "Objaverse_82K",
        save_transformers_folder_path,
        overwrite,
    )
