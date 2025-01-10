import sys
sys.path.append('../distribution-manage/')
sys.path.append('../pointnet-pp/')

from ma_sh.Config.custom_path import toDatasetRootPath
from ma_sh.Method.mash_distribution import clusterAnchors
from ma_sh.Module.anchor_marker import AnchorMarker

if __name__ == "__main__":
    dataset_root_folder_path = toDatasetRootPath()

    if dataset_root_folder_path is None:
        print('[ERROR][get_mash_distribution::__main__]')
        print('\t dataset not found!')
        exit()

    mash_folder_path = dataset_root_folder_path + 'Objaverse_82K/manifold_mash/'
    mash_folder_path = '/home/chli/Dataset_tmp/' + 'Objaverse_82K/manifold_mash/'
    save_feature_folder_path = dataset_root_folder_path + 'Objaverse_82K/anchor_feature/'
    save_feature_folder_path = '/home/chli/Dataset_tmp/' + 'Objaverse_82K/anchor_feature/'
    n_clusters_list = list(range(2, 41))
    save_kmeans_center_npy_folder_path = dataset_root_folder_path + 'Objaverse_82K/kmeans/'
    overwrite = False
    plot_label = False
    plot_error = False

    clusterAnchors(
        mash_folder_path,
        save_feature_folder_path,
        save_kmeans_center_npy_folder_path,
        n_clusters_list,
        overwrite,
        plot_label,
        plot_error,
    )

    exit()

    mash_file_path = '/home/chli/chLi/Dataset/MashV4/ShapeNet/03001627/1006be65e7bc937e9141f9b58470d646.npy'

    anchor_marker = AnchorMarker(
        save_kmeans_center_npy_file_path,
    )

    anchor_marker.markAndRenderMashFile(mash_file_path)
