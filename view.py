import os
import argparse

from ma_sh.Demo.view import demo_view_mash, demo_view_training, demo_view_folder


def isTrainingFolder(mash_folder_path: str) -> bool:
    if mash_folder_path[-1] != "/":
        mash_folder_path += "/"

    mash_filename_list = os.listdir(mash_folder_path)
    for mash_filename in mash_filename_list:
        if "_train" in mash_filename:
            return True

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="view mash")

    parser.add_argument("mash_data_path", help="mash data path")
    args = parser.parse_args()

    mash_data_path = args.mash_data_path

    if os.path.isdir(mash_data_path):
        if isTrainingFolder(mash_data_path):
            demo_view_training(mash_data_path, 1)
        else:
            demo_view_folder(mash_data_path)
    elif mash_data_path.endswith(".npy"):
        demo_view_mash(mash_data_path)
    else:
        print("[ERROR][view::__main__]")
        print("\t mash data path not valid!")
        print("\t mash_data_path:", mash_data_path)
