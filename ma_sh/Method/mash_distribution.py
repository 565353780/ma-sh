import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
from collections import Counter
from typing import Tuple, Union
from multiprocessing import Pool
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from distribution_manage.Module.transformer import Transformer

from ma_sh.Method.path import createFileFolder, removeFile, renameFile
from ma_sh.Method.rotate import toOrthoPosesFromRotateVectors
from ma_sh.Method.transformer import getTransformer


def loadMashFile(mash_file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    params_dict = np.load(mash_file_path, allow_pickle=True).item()
    rotate_vectors = params_dict["rotate_vectors"]
    positions = params_dict["positions"]
    mask_params = params_dict["mask_params"]
    sh_params = params_dict["sh_params"]

    return rotate_vectors, positions, mask_params, sh_params

def loadMashFolder(mash_folder_path: str,
                   keep_dim: bool = False) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[None, None, None, None]]:
    if not os.path.exists(mash_folder_path):
        print('[ERROR][mash_distribution::loadMashFolder]')
        print('\t mash folder not exist!')
        print('\t mash_folder_path:', mash_folder_path)

        return None, None, None, None

    mash_file_path_list = []
    for root, _, files in os.walk(mash_folder_path):
        for file in files:
            if not file.endswith('.npy'):
                continue

            mash_file_path_list.append(root + '/' + file)

    print('[INFO][mash_distribution::loadMashFolder]')
    print('\t start load mash files...')
    with Pool(os.cpu_count()) as pool:
        result_list = list(tqdm(pool.imap(loadMashFile, mash_file_path_list), total=len(mash_file_path_list)))

    if keep_dim:
        rotate_vectors_array = np.stack([result[0] for result in result_list], axis=0)
        positions_array = np.stack([result[1] for result in result_list], axis=0)
        mask_params_array = np.stack([result[2] for result in result_list], axis=0)
        sh_params_array = np.stack([result[3] for result in result_list], axis=0)
    else:
        rotate_vectors_array = np.vstack([result[0] for result in result_list])
        positions_array = np.vstack([result[1] for result in result_list])
        mask_params_array = np.vstack([result[2] for result in result_list])
        sh_params_array = np.vstack([result[3] for result in result_list])

    return rotate_vectors_array, positions_array, mask_params_array, sh_params_array

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
    silhouette_scores = []

    for k in n_clusters_list:
        print('[INFO][mash_distribution::plotKMeansError]')
        print('\t start cluster at n_clusters:', k)
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(anchors)

        sse.append(kmeans.inertia_)
        print('\t\t sse: ', kmeans.inertia_)

        '''
        silhouette_avg = silhouette_score(anchors, kmeans.labels_)
        silhouette_scores.append(silhouette_avg)
        print('silhouette_avg:', silhouette_avg)
        '''

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(n_clusters_list, sse, 'bo-')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('SSE')

    '''
    plt.subplot(1, 2, 2)
    plt.plot(n_clusters_list, silhouette_scores, 'go-')
    plt.title('Silhouette Method For Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    '''

    plt.tight_layout()
    plt.show()

    return True

def clusterAnchors(
    mash_folder_path: str,
    save_kmeans_center_npy_file_path: str,
    n_clusters: int = 4,
    overwrite: bool = False,
    plot_label: bool = False,
    plot_error: bool = False,
) -> bool:
    if os.path.exists(save_kmeans_center_npy_file_path):
        if not overwrite:
            if not plot_error:
                return True

        removeFile(save_kmeans_center_npy_file_path)

    rotate_vectors_array, positions_array, mask_params_array, sh_params_array = loadMashFolder(mash_folder_path, False)
    if rotate_vectors_array is None or positions_array is None or mask_params_array is None or sh_params_array is None:
        print('[ERROR][mash_distribution::clusterAnchors]')
        print('\t loadMashFolder failed!')

        return False

    anchors = sh_params_array
    print('anchors:', anchors.shape)

    if not os.path.exists(save_kmeans_center_npy_file_path):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(anchors)

        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        createFileFolder(save_kmeans_center_npy_file_path)

        tmp_save_kmeans_center_npy_file_path = save_kmeans_center_npy_file_path[:-4] + '_tmp.npy'

        np.save(tmp_save_kmeans_center_npy_file_path, centers)

        renameFile(tmp_save_kmeans_center_npy_file_path, save_kmeans_center_npy_file_path)

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

    if plot_error:
        plotKMeansError(anchors, list(range(2, 41)))

    return True
