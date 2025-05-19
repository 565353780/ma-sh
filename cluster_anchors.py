import sys

sys.path.append("../distribution-manage/")
sys.path.append("../pointnet-pp/")

import os

from ma_sh.Config.custom_path import toDatasetRootPath
from ma_sh.Method.mash_distribution import clusterAnchors
from ma_sh.Module.anchor_marker import AnchorMarker

if __name__ == "__main__":
    dataset_root_folder_path = toDatasetRootPath()

    if dataset_root_folder_path is None:
        print("[ERROR][get_mash_distribution::__main__]")
        print("\t dataset not found!")
        exit()

    mash_folder_path = dataset_root_folder_path + "Objaverse_82K/manifold_mash/"
    mash_folder_path = "/home/chli/Dataset_tmp/" + "Objaverse_82K/manifold_mash/"
    save_feature_folder_path = (
        dataset_root_folder_path + "Objaverse_82K/anchor_feature/"
    )
    save_feature_folder_path = (
        "/home/chli/Dataset_tmp/" + "Objaverse_82K/anchor_feature/"
    )
    n_clusters_list = list(range(2, 41))
    batch_size = 1000
    save_kmeans_center_npy_folder_path = (
        dataset_root_folder_path + "Objaverse_82K/kmeans/b_" + str(batch_size) + "/"
    )
    overwrite = False
    plot_label = False
    plot_error = True

    if plot_error:
        clusterAnchors(
            mash_folder_path,
            save_feature_folder_path,
            save_kmeans_center_npy_folder_path,
            n_clusters_list,
            batch_size,
            overwrite,
            plot_label,
            plot_error,
        )

    mash_file_path = (
        dataset_root_folder_path
        + "MashV4/ShapeNet/03001627/1006be65e7bc937e9141f9b58470d646.npy"
    )
    color_map = "mkl"
    device = "cuda:0"

    anchor_marker = AnchorMarker(
        save_kmeans_center_npy_folder_path + str(5) + ".npy",
        color_map,
        device,
    )

    # anchor_marker.markAndRenderMashFile(mash_file_path)
    # anchor_marker.markAndRenderAnchorClusters(mash_file_path, cluster_dist=0.1)
    # anchor_marker.markAndRenderAverageAnchors(mash_file_path)
    # anchor_marker.markAndRenderMashReplacedByAverageAnchors(mash_file_path)

    # anchor_marker.markAndRenderMashFile(
    # '/home/chli/chLi/Dataset/Objaverse_82K/manifold_mash/000-091/6fee90b9a77c482c9928771f03287733.npy')

    mash_folder_path = dataset_root_folder_path + "Objaverse_82K/manifold_mash/"

    for root, _, files in os.walk(mash_folder_path):
        for file in files:
            if not file.endswith(".npy") or file.endswith("_tmp.npy"):
                continue

            mash_file_path = root + "/" + file

            print("mash:", mash_file_path)
            anchor_marker.markAndRenderMashFile(mash_file_path)
            anchor_marker.markAndRenderAnchorClusters(mash_file_path, cluster_dist=0.1)
