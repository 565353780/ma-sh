import os

from ma_sh.Config.custom_path import toDatasetRootPath
from ma_sh.Method.path import removeFile

def clearTag(tag_folder_path: str) -> bool:
    for root, _, files in os.walk(tag_folder_path):
        for file in files:
            if not file.endswith('start.txt'):
                continue

            removeFile(root + '/' + file)

    return True

if __name__ == "__main__":
    dataset_root_folder_path = toDatasetRootPath()
    if dataset_root_folder_path is None:
        print('[ERROR][clear_tag::__main__]')
        print('\t dataset not found!')
        exit()

    pcd_tag_folder_path = dataset_root_folder_path + 'Objaverse_82K/'

    clearTag(pcd_tag_folder_path)
