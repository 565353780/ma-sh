import os
import pickle
import numpy as np
from time import time

from ma_sh.Method.path import createFileFolder, renameFile, removeFile


def clearTag(
    tag_folder_path: str,
    file_format: str,
    dry_run: bool = False,
    output_freq: float = 1.0,
) -> bool:
    solved_shape_num = 0
    cleared_tag_num = 0

    start = time()
    for root, _, files in os.walk(tag_folder_path):
        for file in files:
            solved_shape_num += 1

            if time() - start >= output_freq:
                print('solved shape num:', solved_shape_num)
                start = time()

            if file.endswith(file_format) and '_tmp' not in file:
                continue

            if not dry_run:
                removeFile(root + "/" + file)

            cleared_tag_num += 1
            print(root + "/" + file)
            print("cleared tag num:", cleared_tag_num)

    return True

def removeInvalidNpy(
    tag_folder_path: str,
    dry_run: bool = False,
    output_freq: float = 1.0,
) -> bool:
    solved_shape_num = 0
    cleared_tag_num = 0

    start = time()
    for root, _, files in os.walk(tag_folder_path):
        for file in files:
            solved_shape_num += 1

            if time() - start >= output_freq:
                print('solved shape num:', solved_shape_num)
                start = time()

            if not file.endswith('.npy') or '_tmp' in file:
                continue

            try:
                np.load(root + '/' + file, allow_pickle=True).item()
            except:
                if not dry_run:
                    removeFile(root + "/" + file)

                cleared_tag_num += 1
                print(root + "/" + file)
                print("cleared tag num:", cleared_tag_num)

    return True



def createDatasetJson(
    source_root_folder_path: str,
    target_root_folder_path: str,
    save_json_file_path: str,
    overwrite: bool = False,
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

    for root, _, files in os.walk(source_root_folder_path):
        for file in files:
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

                target_file_path = target_folder_path + target_file_name

                try:
                    data = np.load(target_file_path, allow_pickle=True).item()
                    assert 'dino' in data.keys()
                except:
                    print('[ERROR][dataset::createDatasetJson]')
                    print('\t load target file failed!')
                    print('\t target_file_path:', target_file_path)
                    continue

                valid_rel_target_file_path_list.append(rel_base_path + '/' + target_file_name)

            if len(valid_rel_target_file_path_list) == 0:
                continue

            valid_rel_target_file_path_list.sort()

            paths_list.append([rel_folder_path + '/' + file, valid_rel_target_file_path_list])

            solved_shape_num += 1
            print("solved shape num:", solved_shape_num)

    paths_list.sort(key=lambda x: x[0])

    createFileFolder(save_json_file_path)

    tmp_save_json_file_path = save_json_file_path[:-5] + '_tmp.json'

    with open(tmp_save_json_file_path, 'wb') as f:
        pickle.dump(paths_list, f)

    renameFile(tmp_save_json_file_path, save_json_file_path)

    print(paths_list)
    print(len(paths_list))
    return True
