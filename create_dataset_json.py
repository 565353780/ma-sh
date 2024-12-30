from ma_sh.Config.custom_path import toDatasetRootPath
from ma_sh.Method.dataset import createDatasetJson

if __name__ == "__main__":
    dataset_root_folder_path = toDatasetRootPath()
    assert dataset_root_folder_path is not None

    task_id = 'mash+render_dino'
    source_root_folder_path = dataset_root_folder_path + 'Objaverse_82K/manifold_mash/'
    target_root_folder_path = dataset_root_folder_path + 'Objaverse_82K/render_dino/'
    save_json_file_path = dataset_root_folder_path + 'Objaverse_82K/render_dino.json'
    overwrite = True
    output_freq = 1.0

    createDatasetJson(
        task_id,
        source_root_folder_path,
        target_root_folder_path,
        save_json_file_path,
        overwrite,
        output_freq,
    )
