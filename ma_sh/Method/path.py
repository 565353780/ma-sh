import os
from time import time
from shutil import rmtree


def createFileFolder(file_path):
    file_name = file_path.split("/")[-1]
    file_folder_path = file_path.split("/" + file_name)[0] + "/"
    os.makedirs(file_folder_path, exist_ok=True)
    return True

def removeFile(file_path):
    while os.path.exists(file_path):
        try:
            os.remove(file_path)
        except:
            continue
    return True

def removeFolder(folder_path):
    while os.path.exists(folder_path):
        try:
            rmtree(folder_path)
        except:
            pass

    return True

def renameFile(source_file_path, target_file_path, overwrite: bool = False):
    if os.path.exists(target_file_path):
        if not overwrite:
            return True

        removeFile(target_file_path)

    while os.path.exists(source_file_path):
        try:
            os.rename(source_file_path, target_file_path)
        except:
            pass
    return True

def renameFolder(source_folder_path: str, target_folder_path: str, overwrite: bool = False):
    if os.path.exists(target_folder_path):
        if not overwrite:
            return True

        removeFolder(target_folder_path)

    while os.path.exists(source_folder_path):
        try:
            os.rename(source_folder_path, target_folder_path)
        except:
            pass

    return True

def waitFile(file_path: str, wait_second: int) -> bool:
    start = time()

    while not os.path.exists(file_path):
        spend = time() - start

        if spend > wait_second:
            break

    return os.path.exists(file_path)
