import os
import pickle
import numpy as np
from time import time
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool

from ma_sh.Method.path import createFileFolder, renameFile, removeFile


def clearTagWithPool(inputs: list) -> bool:
    file_path, dry_run = inputs

    if dry_run:
        print('tag file:', file_path)
    else:
        removeFile(file_path)

    return True

def clearTag(
    tag_folder_path: str,
    file_format: str,
    dry_run: bool = False,
    worker_num: int = os.cpu_count(),
) -> bool:
    inputs_list = []

    for root, _, files in os.walk(tag_folder_path):
        for file in files:

            if file.endswith(file_format) and '_tmp' not in file:
                continue

            inputs_list.append([root + '/' + file, dry_run])

    print('[INFO][dataset::clearTag]')
    print('\t start remove tag files...')
    try:
        with Pool(worker_num) as pool:
            results = list(tqdm(pool.imap(clearTagWithPool, inputs_list), total=len(inputs_list)))
    except RuntimeError as e:
        print('[ERROR][dataset::clearTag]')
        print('\t main process caught an exception:', e)
        exit()

    return True

def removeInvalidPNGWithPool(inputs: list) -> bool:
    check_file_path, dry_run = inputs

    try:
        with Image.open(check_file_path) as img:
            img.verify()
    except KeyboardInterrupt:
        print('[INFO][dataset::removeInvalidPNGWithPool]')
        print('\t program interrupted by the user (Ctrl+C).')
        exit()
    except:
        if dry_run:
            print('invalid file:', check_file_path)
        else:
            removeFile(check_file_path)

    return True

def removeInvalidPNG(
    tag_folder_path: str,
    dry_run: bool = False,
    worker_num: int = os.cpu_count(),
) -> bool:
    inputs_list = []

    for root, _, files in os.walk(tag_folder_path):
        for file in files:
            if not file.endswith('.png') or '_tmp' in file:
                continue

            inputs_list.append([root + '/' + file, dry_run])

    print('[INFO][dataset::removeInvalidPNG]')
    print('\t start remove invalid png files...')
    try:
        with Pool(worker_num) as pool:
            results = list(tqdm(pool.imap(removeInvalidPNGWithPool, inputs_list), total=len(inputs_list)))
    except RuntimeError as e:
        print('[ERROR][dataset::removeInvalidPNG]')
        print('\t main process caught an exception:', e)
        exit()

    return True

def removeInvalidNPYWithPool(inputs: list) -> bool:
    check_file_path, dry_run = inputs

    try:
        np.load(check_file_path, allow_pickle=True).item()
    except KeyboardInterrupt:
        print('[INFO][dataset::removeInvalidNPYWithPool]')
        print('\t program interrupted by the user (Ctrl+C).')
        exit()
    except:
        if dry_run:
            print('invalid file:', check_file_path)
        else:
            removeFile(check_file_path)

    return True

def removeInvalidNPY(
    tag_folder_path: str,
    dry_run: bool = False,
    worker_num: int = os.cpu_count(),
) -> bool:
    inputs_list = []

    for root, _, files in os.walk(tag_folder_path):
        for file in files:
            if not file.endswith('.npy') or '_tmp' in file:
                continue

            inputs_list.append([root + '/' + file, dry_run])

    print('[INFO][dataset::removeInvalidNPY]')
    print('\t start remove invalid npy files...')
    try:
        with Pool(worker_num) as pool:
            results = list(tqdm(pool.imap(removeInvalidNPYWithPool, inputs_list), total=len(inputs_list)))
    except RuntimeError as e:
        print('[ERROR][dataset::removeInvalidNPY]')
        print('\t main process caught an exception:', e)
        exit()

    return True

def createDatasetJson(
    task_id: str,
    source_root_folder_path: str,
    target_root_folder_path: str,
    save_json_file_path: str,
    overwrite: bool = False,
    output_freq: float = 1.0,
) -> bool:
    if not os.path.exists(source_root_folder_path):
        print('[ERROR][dataset::createDatasetJson]')
        print('\t source root folder not found!')
        print('\t source_root_folder_path:', source_root_folder_path)
        return False

    if not os.path.exists(target_root_folder_path):
        print('[ERROR][dataset::createDatasetJson]')
        print('\t target root folder not found!')
        print('\t target_root_folder_path:', target_root_folder_path)
        return False

    if os.path.exists(save_json_file_path):
        if not overwrite:
            return True

    print("[INFO][BaseConvertor::convertAll]")
    print("\t start convert all data...")
    solved_shape_num = 0

    paths_list = []

    start = time()
    for root, _, files in os.walk(source_root_folder_path):
        for file in files:
            solved_shape_num += 1

            if time() - start >= output_freq:
                print('[' + task_id + "] solved shape num:", solved_shape_num)
                start = time()

            if not file.endswith('.npy'):
                continue

            if file.endswith('_tmp.npy'):
                continue

            rel_folder_path = os.path.relpath(root, source_root_folder_path)

            rel_base_path = rel_folder_path + '/' + file[:-4]

            target_folder_path = target_root_folder_path + rel_base_path + '/'

            if not os.path.exists(target_folder_path):
                continue

            target_file_name_list = os.listdir(target_folder_path)

            valid_rel_target_file_path_list = []

            for target_file_name in target_file_name_list:
                if not target_file_name.endswith('.npy'):
                    continue

                if target_file_name.endswith('_tmp.npy'):
                    continue

                '''
                target_file_path = target_folder_path + target_file_name

                try:
                    data = np.load(target_file_path, allow_pickle=True).item()
                    assert 'dino' in data.keys()
                except KeyboardInterrupt:
                    print('[INFO][dataset::removeInvalidNPY]')
                    print('\t program interrupted by the user (Ctrl+C).')
                    exit()
                except:
                    print('[ERROR][dataset::createDatasetJson]')
                    print('\t load target file failed!')
                    print('\t target_file_path:', target_file_path)
                    continue
                '''

                valid_rel_target_file_path_list.append(rel_base_path + '/' + target_file_name)

            if len(valid_rel_target_file_path_list) == 0:
                continue

            valid_rel_target_file_path_list.sort()

            paths_list.append([rel_folder_path + '/' + file, valid_rel_target_file_path_list])

    paths_list.sort(key=lambda x: x[0])

    createFileFolder(save_json_file_path)

    tmp_save_json_file_path = save_json_file_path[:-5] + '_tmp.json'

    with open(tmp_save_json_file_path, 'wb') as f:
        pickle.dump(paths_list, f)

    renameFile(tmp_save_json_file_path, save_json_file_path)

    print(len(paths_list))
    return True
