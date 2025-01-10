import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Union
from multiprocessing import Pool

from pointnet_pp.Module.detector import Detector

from ma_sh.Model.simple_mash import SimpleMash
from ma_sh.Method.io import loadMashFolder


def toFeatureFiles(
    mash_folder_path: str,
    save_feature_folder_path: str,
    batch_size: int = 800,
    save_freq: int = 100,
    overwrite: bool = False,
) -> bool:
    os.makedirs(save_feature_folder_path, exist_ok=True)

    rotate_vectors_array, positions_array, mask_params_array, sh_params_array = loadMashFolder(mash_folder_path, False)
    if rotate_vectors_array is None or positions_array is None or mask_params_array is None or sh_params_array is None:
        print('[ERROR][feature::toFeatureFiles]')
        print('\t loadMashFolder failed!')

        return False

    rotate_vectors_array = np.zeros_like(rotate_vectors_array)
    positions_array = np.zeros_like(positions_array)

    model_file_path = "../pointnet-pp/pretrained/cls_ssg/best_model.pth"
    device = "cuda:0"
    detector = None

    chunk_num = int(rotate_vectors_array.shape[0] / batch_size)

    last_batch_size = rotate_vectors_array.shape[0] - chunk_num * batch_size

    features = []

    mash = SimpleMash(
        batch_size,
        sample_phi_num=20,
        sample_theta_num=20,
        device='cuda:0')

    pbar = tqdm(total=rotate_vectors_array.shape[0])
    for i in range(chunk_num):
        current_id = int(i / save_freq)

        current_save_npy_file_path = save_feature_folder_path + str(current_id) + '.npy'

        if not overwrite:
            if os.path.exists(current_save_npy_file_path):
                pbar.update(batch_size)
                continue

        batch_r = rotate_vectors_array[i * batch_size: (i + 1) * batch_size, :]
        batch_p = positions_array[i * batch_size: (i + 1) * batch_size, :]
        batch_m = mask_params_array[i * batch_size: (i + 1) * batch_size, :]
        batch_s = sh_params_array[i * batch_size: (i + 1) * batch_size, :]

        mash.loadParams(batch_m, batch_s, batch_r, batch_p)

        inner_pts = mash.toSamplePoints()[1]
        inner_pts = inner_pts.reshape(batch_size, -1, 3).permute(0, 2, 1)

        if detector is None:
            detector = Detector(model_file_path, device)

        _, feature = detector.detect(inner_pts)
        features.append(feature)

        if (i + 1) % save_freq == 0:
            features_array = torch.vstack(features).squeeze(-1).cpu().numpy()

            np.save(current_save_npy_file_path, features_array)

            features = []

        pbar.update(batch_size)

    current_id = int(chunk_num / save_freq)

    current_save_npy_file_path = save_feature_folder_path + str(current_id) + '.npy'

    if last_batch_size > 0:
        if not overwrite:
            if os.path.exists(current_save_npy_file_path):
                pbar.update(last_batch_size)
                pbar.close()
                return True

        mash = SimpleMash(
            last_batch_size,
            sample_phi_num=10,
            sample_theta_num=10,
            device='cuda:0')

        batch_r = rotate_vectors_array[-last_batch_size:]
        batch_p = positions_array[-last_batch_size:]
        batch_m = mask_params_array[-last_batch_size:]
        batch_s = sh_params_array[-last_batch_size:]

        mash.loadParams(batch_m, batch_s, batch_r, batch_p)

        inner_pts = mash.toSamplePoints()[1]
        inner_pts = inner_pts.reshape(last_batch_size, -1, 3).permute(0, 2, 1)

        if detector is None:
            detector = Detector(model_file_path, device)

        _, feature = detector.detect(inner_pts)
        features.append(feature)

        pbar.update(last_batch_size)

    if len(features) > 0:
        features_array = torch.vstack(features).squeeze(-1).cpu().numpy()

        np.save(current_save_npy_file_path, features_array)

    pbar.close()

    return True

def loadFeatures(
    mash_folder_path: str,
    save_feature_folder_path: str,
    batch_size: int = 800,
    save_freq: int = 100,
    worker_num: int = 1,
    overwrite: bool = False,
    return_file_path_list_only: bool = False,
) -> Union[np.ndarray, list, None]:
    if not toFeatureFiles(
        mash_folder_path,
        save_feature_folder_path,
        batch_size,
        save_freq,
        overwrite,
    ):
        print('[ERROR][feature::loadFeatures]')
        print('\t toFeatureFiles failed!')
        return None

    feature_filename_list = os.listdir(save_feature_folder_path)

    valid_feature_file_path_list = []

    for feature_filename in feature_filename_list:
        if not feature_filename.endswith('.npy'):
            continue

        valid_feature_file_path_list.append(save_feature_folder_path + feature_filename)

    if return_file_path_list_only:
        return valid_feature_file_path_list

    print('[INFO][feature::loadFeatures]')
    print('\t start load featrue files...')
    with Pool(worker_num) as pool:
        features = list(tqdm(pool.imap(np.load, valid_feature_file_path_list), total=len(valid_feature_file_path_list)))

    features_array = np.vstack(features)

    return features_array
