from ma_sh.Config.custom_path import toDatasetRootPath
from ma_sh.Method.dataset import createDatasetPickle

if __name__ == "__main__":
    dataset_root_folder_path = toDatasetRootPath()
    assert dataset_root_folder_path is not None

    task_id = "mash+render_dino"
    source_root_folder_path = dataset_root_folder_path + "Objaverse_82K/manifold_mash/"
    target_root_folder_path = dataset_root_folder_path + "Objaverse_82K/render_dino/"
    save_pickle_file_path = dataset_root_folder_path + "Objaverse_82K/render_dino.pkl"
    overwrite = True
    output_freq = 1.0

    createDatasetPickle(
        task_id,
        source_root_folder_path,
        target_root_folder_path,
        save_pickle_file_path,
        overwrite,
        output_freq,
    )
