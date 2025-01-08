import os
import torch
import numpy as np
from tqdm import tqdm

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
    rotate_vectors_array, positions_array, mask_params_array, sh_params_array = loadMashFolder(mash_folder_path, False)
    if rotate_vectors_array is None or positions_array is None or mask_params_array is None or sh_params_array is None:
        print('[ERROR][feature::toFeatureFiles]')
        print('\t loadMashFolder failed!')

        return False

    model_file_path = "../pointnet-pp/pretrained/cls_ssg/best_model.pth"
    device = "cuda:0"
    detector = Detector(model_file_path, device)

    chunk_num = int(rotate_vectors_array.shape[0] / batch_size)

    last_batch_size = rotate_vectors_array.shape[0] - chunk_num * batch_size

    features = []

    mash = SimpleMash(
        batch_size,
        sample_phi_num=10,
        sample_theta_num=10,
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

        _, feature = detector.detect(inner_pts)
        features.append(feature)

        if (i + 1) % save_freq == 0:
            features_array = torch.vstack(features).squeeze(-1).cpu().numpy()

            np.save(current_save_npy_file_path, features_array)

            features = []

        pbar.update(batch_size)

    if last_batch_size > 0:
        current_id = int(chunk_num / save_freq)

        current_save_npy_file_path = save_feature_folder_path + str(current_id) + '.npy'

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

        _, feature = detector.detect(inner_pts)
        features.append(feature)

        features_array = torch.vstack(features).squeeze(-1).cpu().numpy()

        np.save(current_save_npy_file_path, features_array)

        pbar.update(last_batch_size)

    pbar.close()

    return True
