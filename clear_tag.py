import os

from ma_sh.Config.custom_path import toDatasetRootPath
from ma_sh.Method.dataset import clearTag, removeInvalidPNG, removeInvalidNPY



if __name__ == "__main__":
    dataset_root_folder_path = toDatasetRootPath()
    clear_tag = True
    remove_invalid_npy = True
    dry_run = False
    worker_num = os.cpu_count()

    if dataset_root_folder_path is None:
        print("[ERROR][clear_tag::__main__]")
        print("\t dataset not found!")
        exit()

    if clear_tag:
        clearTag(dataset_root_folder_path + "Objaverse_82K/glbs/", '.glb', dry_run, worker_num)
        clearTag(dataset_root_folder_path + "Objaverse_82K/mesh/", '.obj', dry_run, worker_num)
        clearTag(dataset_root_folder_path + "Objaverse_82K/manifold/", '.obj', dry_run, worker_num)
        clearTag(dataset_root_folder_path + "Objaverse_82K/manifold_pcd/", '.npy', dry_run, worker_num)
        clearTag(dataset_root_folder_path + "Objaverse_82K/manifold_mash/", '.npy', dry_run, worker_num)
        clearTag(dataset_root_folder_path + "Objaverse_82K/render/", '.png', dry_run, worker_num)
        clearTag(dataset_root_folder_path + "Objaverse_82K/render_dino/", '.npy', dry_run, worker_num)
        clearTag(dataset_root_folder_path + "Objaverse_82K/manifold_sdf_0_25/", '.npy', dry_run, worker_num)

    if remove_invalid_npy:
        removeInvalidPNG(dataset_root_folder_path + "Objaverse_82K/render/", dry_run, worker_num)
        removeInvalidNPY(dataset_root_folder_path + "Objaverse_82K/render_dino/", dry_run, worker_num)
