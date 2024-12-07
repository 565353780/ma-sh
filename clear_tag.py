import os
from shutil import rmtree

def clearTag(tag_folder_path: str) -> bool:
    remove_folder_path_list = []

    for root, _, files in os.walk(tag_folder_path):
        for file in files:
            if file != 'start.txt':
                continue

            if not os.path.exists(root + '/finish.txt'):
                remove_folder_path_list.append(root + '/')

    for remove_folder_path in remove_folder_path_list:
        rmtree(remove_folder_path)

    return True

if __name__ == "__main__":
    pcd_tag_folder_path = os.environ['HOME'] + '/chLi/Dataset/Tag/Objaverse_82K_pcd/'

    clearTag(pcd_tag_folder_path)
