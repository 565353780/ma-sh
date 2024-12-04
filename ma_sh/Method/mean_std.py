import os
import numpy as np
from tqdm import tqdm
from typing import Tuple, Union
from multiprocessing import Pool


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
        print('[ERROR][mean_std::loadMashFolder]')
        print('\t mash folder not exist!')
        print('\t mash_folder_path:', mash_folder_path)

        return None, None, None, None

    mash_file_path_list = []
    for root, _, files in os.walk(mash_folder_path):
        for file in files:
            if not file.endswith('.npy'):
                continue

            mash_file_path_list.append(root + '/' + file)

    print('[INFO][mean_std::loadMashFolder]')
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

def getMashMeanAndSTD(mash_folder_path: str) -> bool:
    rotate_vectors_array, positions_array, mask_params_array, sh_params_array = loadMashFolder(mash_folder_path, False)
    if rotate_vectors_array is None or positions_array is None or mask_params_array is None or sh_params_array is None:
        print('[ERROR][mean_std::getMashMeanAndSTD]')
        print('\t loadMashFolder failed!')

        return False

    rotate_vectors_mean = np.mean(rotate_vectors_array, axis=0)
    rotate_vectors_std = np.std(rotate_vectors_array, axis=0)
    positions_mean = np.mean(positions_array, axis=0)
    positions_std = np.std(positions_array, axis=0)
    mask_params_mean = np.mean(mask_params_array, axis=0)
    mask_params_std = np.std(mask_params_array, axis=0)
    sh_params_mean = np.mean(sh_params_array, axis=0)
    sh_params_std = np.std(sh_params_array, axis=0)

    print('[INFO][mean_std::getMashMeanAndSTD]')
    outputArray('ROTATE_VECTORS_MEAN', rotate_vectors_mean)
    outputArray('ROTATE_VECTORS_STD', rotate_vectors_std)
    outputArray('POSITIONS_MEAN', positions_mean)
    outputArray('POSITIONS_STD', positions_std)
    outputArray('MASK_PARAMS_MEAN', mask_params_mean)
    outputArray('MASK_PARAMS_STD', mask_params_std)
    outputArray('SH_PARAMS_MEAN', sh_params_mean)
    outputArray('SH_PARAMS_STD', sh_params_std)

    return True
