from ma_sh.Config.custom_path import toDatasetRootPath
from ma_sh.Method.dataset import clearTag



if __name__ == "__main__":
    dataset_root_folder_path = toDatasetRootPath()
    dry_run = False

    if dataset_root_folder_path is None:
        print("[ERROR][clear_tag::__main__]")
        print("\t dataset not found!")
        exit()

    clearTag(dataset_root_folder_path + "Objaverse_82K/glbs/", '.glb', dry_run)
    clearTag(dataset_root_folder_path + "Objaverse_82K/mesh/", '.obj', dry_run)
    clearTag(dataset_root_folder_path + "Objaverse_82K/manifold/", '.obj', dry_run)
    clearTag(dataset_root_folder_path + "Objaverse_82K/manifold_pcd/", '.npy', dry_run)
    clearTag(dataset_root_folder_path + "Objaverse_82K/manifold_mash/", '.npy', dry_run)
    clearTag(dataset_root_folder_path + "Objaverse_82K/render/", '.png', dry_run)
    clearTag(dataset_root_folder_path + "Objaverse_82K/render_dino/", '.npy', dry_run)
    clearTag(dataset_root_folder_path + "Objaverse_82K/manifold_sdf_0_25/", '.npy', dry_run)
