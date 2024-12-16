import os
from pathlib import Path
from typing import Union
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

def isHaveSubFolder(folder_path: Union[Path, str]) -> bool:
    if isinstance(folder_path, str):
        folder = Path(folder_path)
    else:
        folder = folder_path

    if not folder.is_dir():
        print('[ERROR][path::isHaveSubFolder]')
        print('\t folder not exist!')
        print('\t folder_path:', folder_path)
        return False

    try:
        return any(item.is_dir() for item in folder.iterdir())
    except PermissionError:
        print('[ERROR][path::isHaveSubFolder]')
        print('\t permission denied!')
        print('\t folder_path:', folder_path)
        return False
