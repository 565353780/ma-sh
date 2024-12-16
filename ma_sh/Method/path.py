import os


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
