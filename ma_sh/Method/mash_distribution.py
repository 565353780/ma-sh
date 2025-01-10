import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
from typing import Union
from collections import Counter
from sklearn.cluster import MiniBatchKMeans

from distribution_manage.Module.transformer import Transformer

from ma_sh.Method.io import loadMashFolder
from ma_sh.Method.feature import loadFeatures
from ma_sh.Method.path import createFileFolder, removeFile, renameFile
from ma_sh.Method.rotate import toOrthoPosesFromRotateVectors


def outputArray(array_name: str, array_value: np.ndarray) -> bool:
    print(array_name, '= [', end='')
    for i in range(array_value.shape[0] - 1):
        print(str(array_value[i]) + ', ', end='')
    print(str(array_value[-1]) + ']')
    return True

def plot_overall_histograms(data, bins=10, save_image_file_path: Union[str, None]=None, render: bool = True):
    num_dimensions = data.shape[1]
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    axes = axes.flatten()

    for i in range(num_dimensions):
        ax = axes[i]
        ax.hist(data[:, i], bins=bins, alpha=0.75, color='blue', edgecolor='black')
        ax.set_title(f"Dimension {i+1}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")

    for j in range(num_dimensions, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    if render:
        plt.show()

    if save_image_file_path is not None:
        createFileFolder(save_image_file_path)

        plt.savefig(save_image_file_path, dpi=300)

    plt.close()
    return True

def getMashDistribution(
    mash_folder_path: str,
    transformers_id: str,
    save_transformers_folder_path: str,
    overwrite: bool = False,
) -> bool:
    save_transformers_file_basepath = save_transformers_folder_path + transformers_id

    save_transformers_file_path = save_transformers_file_basepath + '.pkl'

    if os.path.exists(save_transformers_file_path):
        if not overwrite:
            return True

        removeFile(save_transformers_file_path)

    os.makedirs(save_transformers_folder_path, exist_ok=True)

    rotate_vectors_array, positions_array, mask_params_array, sh_params_array = loadMashFolder(mash_folder_path, False)
    if rotate_vectors_array is None or positions_array is None or mask_params_array is None or sh_params_array is None:
        print('[ERROR][mash_distribution::getMashDistribution]')
        print('\t loadMashFolder failed!')

        return False

    print('start toOrthoPosesFromRotateVectors...')
    rotations_array = toOrthoPosesFromRotateVectors(torch.from_numpy(rotate_vectors_array).to(torch.float64)).numpy()

    data = np.hstack([rotations_array, positions_array, mask_params_array, sh_params_array])

    Transformer.plotDistribution(data, 100, save_transformers_file_basepath + '_vis_data.pdf', False)

    # Transformer.fit('uniform', data, './output/uniform_transformers.pkl', False)
    # Transformer.fit('normal', data, './output/normal_transformers.pkl', False)
    # Transformer.fit('power', data, './output/power_transformers.pkl', False)
    # Transformer.fit('robust', data, './output/robust_scalers.pkl', False)
    # Transformer.fit('binary', data, './output/binarizers.pkl', False)
    # Transformer.fit('kernel', data, './output/kernel_centerers.pkl', False)
    # Transformer.fit('min_max', data, './output/min_max_scalers.pkl', False)
    # Transformer.fit('max_abs', data, './output/max_abs_scalers.pkl', False)
    # Transformer.fit('standard', data, './output/standard_scalers.pkl', False)
    Transformer.fit('multi_linear', data, save_transformers_file_path, False)

    transformer = Transformer(save_transformers_file_path)

    print('start transformData...')
    start = time()
    trans_data = transformer.transform(data)
    print('transform time:', time() - start)

    Transformer.plotDistribution(trans_data, 100, save_transformers_file_basepath + '_vis_trans_data.pdf', False)

    print('start transformData with inverse...')
    start = time()
    trans_back_data = transformer.inverse_transform(trans_data)
    print('inverse_transform time:', time() - start)

    Transformer.plotDistribution(trans_back_data, 100, save_transformers_file_basepath + '_vis_trans_back_data.pdf', False)

    error_max = np.max(np.abs(data - trans_back_data))

    print('error_max =', error_max)

    return True

    rotations_mean = np.mean(rotations_array, axis=0)
    rotations_std = np.std(rotations_array, axis=0)
    positions_mean = np.mean(positions_array, axis=0)
    positions_std = np.std(positions_array, axis=0)
    mask_params_mean = np.mean(mask_params_array, axis=0)
    mask_params_std = np.std(mask_params_array, axis=0)
    sh_params_mean = np.mean(sh_params_array, axis=0)
    sh_params_std = np.std(sh_params_array, axis=0)

    print('[INFO][mash_distribution::getMashMeanAndSTD]')
    outputArray('ROTATIONS_MEAN', rotations_mean)
    outputArray('ROTATIONS_STD', rotations_std)
    outputArray('POSITIONS_MEAN', positions_mean)
    outputArray('POSITIONS_STD', positions_std)
    outputArray('MASK_PARAMS_MEAN', mask_params_mean)
    outputArray('MASK_PARAMS_STD', mask_params_std)
    outputArray('SH_PARAMS_MEAN', sh_params_mean)
    outputArray('SH_PARAMS_STD', sh_params_std)

    return True

def plotKMeansError(anchors: np.ndarray, n_clusters_list: list) -> bool:
    sse = []

    for k in n_clusters_list:
        print('[INFO][mash_distribution::plotKMeansError]')
        print('\t start cluster at n_clusters:', k)
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(anchors)

        sse.append(kmeans.inertia_)
        print('\t\t sse: ', kmeans.inertia_)

    plt.figure(figsize=(12, 6))
    plt.plot(n_clusters_list, sse, 'bo-')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('SSE')

    plt.tight_layout()
    plt.show()

    return True

def clusterAnchorsStep(
    feature_file_path_list: list,
    save_kmeans_center_npy_file_path: str,
    n_clusters: int = 4,
    overwrite: bool = False,
    plot_label: bool = False,
) -> bool:
    if not overwrite:
        if os.path.exists(save_kmeans_center_npy_file_path):
            return True

        removeFile(save_kmeans_center_npy_file_path)

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        max_iter=300,
        batch_size=1000,
        max_no_improvement=20,
        random_state=0,
    )

    print('[INFO][mash_distribution::clusterAnchorsStep]')
    print('\t start partial fitting features with n_clusters =', n_clusters, '...')
    for feature_file_path in tqdm(feature_file_path_list):
        features = np.load(feature_file_path)
        kmeans.partial_fit(features)

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    createFileFolder(save_kmeans_center_npy_file_path)

    tmp_save_kmeans_center_npy_file_path = save_kmeans_center_npy_file_path[:-4] + '_tmp.npy'

    np.save(tmp_save_kmeans_center_npy_file_path, centers)

    renameFile(tmp_save_kmeans_center_npy_file_path, save_kmeans_center_npy_file_path)

    save_inertia_txt_file_path = save_kmeans_center_npy_file_path[:-4] + '_inertia.txt'
    with open(save_inertia_txt_file_path, 'w') as f:
        f.write(str(kmeans.inertia_))

    if plot_label:
        cluster_counts = Counter(labels)

        plt.bar(cluster_counts.keys(), cluster_counts.values(), color='skyblue')
        plt.title('Cluster Size Distribution')
        plt.xlabel('Cluster Label')
        plt.ylabel('Number of Points')
        plt.xticks(range(kmeans.n_clusters))
        plt.show()

        plt.pie(cluster_counts.values(), labels=cluster_counts.keys(), autopct='%1.1f%%', colors=plt.cm.tab10.colors)
        plt.title('Cluster Size Proportion')
        plt.show()

    return True

def clusterAnchors(
    mash_folder_path: str,
    save_feature_folder_path: str,
    save_kmeans_center_npy_folder_path: str,
    n_clusters_list: list = list(range(2, 41)),
    overwrite: bool = False,
    plot_label: bool = False,
    plot_error: bool = False,
) -> bool:
    os.makedirs(save_kmeans_center_npy_folder_path, exist_ok=True)

    feature_file_path_list = loadFeatures(
        mash_folder_path,
        save_feature_folder_path,
        800,
        100,
        4,
        overwrite,
        True,
    )
    assert isinstance(feature_file_path_list, list)

    if len(feature_file_path_list) == 0:
        print('[ERROR][mash_distribution::clusterAnchors]')
        print('\t feature file not found!')
        return False

    for n_clusters in n_clusters_list:
        save_kmeans_center_npy_file_path = save_kmeans_center_npy_folder_path + str(n_clusters) + '.npy'

        if not clusterAnchorsStep(
            feature_file_path_list,
            save_kmeans_center_npy_file_path,
            n_clusters,
            overwrite,
            plot_label
        ):
            print('[WARN][mash_distribution::clusterAnchors]')
            print('\t clusterAnchorsStep failed!')
            print('\t n_clusters:', n_clusters)
            continue

    if plot_error:
        plotKMeansError(feature_file_path_list, list(range(2, 41)))

    return True
