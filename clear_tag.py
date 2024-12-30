from ma_sh.Config.custom_path import toDatasetRootPath
from ma_sh.Method.dataset import clearTag, removeInvalidPNG, removeInvalidNPY



if __name__ == "__main__":
    dataset_root_folder_path = toDatasetRootPath()
    clear_tag = True
    remove_invalid_npy = True
    dry_run = False
    output_freq = 1.0

    if dataset_root_folder_path is None:
        print("[ERROR][clear_tag::__main__]")
        print("\t dataset not found!")
        exit()

    if clear_tag:
        clearTag('glbs', dataset_root_folder_path + "Objaverse_82K/glbs/", '.glb', dry_run, output_freq)
        clearTag('mesh', dataset_root_folder_path + "Objaverse_82K/mesh/", '.obj', dry_run, output_freq)
        clearTag('manifold', dataset_root_folder_path + "Objaverse_82K/manifold/", '.obj', dry_run, output_freq)
        clearTag('manifold_pcd', dataset_root_folder_path + "Objaverse_82K/manifold_pcd/", '.npy', dry_run, output_freq)
        clearTag('manifold_mash', dataset_root_folder_path + "Objaverse_82K/manifold_mash/", '.npy', dry_run, output_freq)
        clearTag('render', dataset_root_folder_path + "Objaverse_82K/render/", '.png', dry_run, output_freq)
        clearTag('render_dino', dataset_root_folder_path + "Objaverse_82K/render_dino/", '.npy', dry_run, output_freq)
        clearTag('manifold_sdf_0_25', dataset_root_folder_path + "Objaverse_82K/manifold_sdf_0_25/", '.npy', dry_run, output_freq)

    if remove_invalid_npy:
        removeInvalidPNG('render_png_check', dataset_root_folder_path + "Objaverse_82K/render/", dry_run, output_freq)
        removeInvalidNPY('render_dino_npy_check', dataset_root_folder_path + "Objaverse_82K/render_dino/", dry_run, output_freq)
