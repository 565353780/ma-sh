import os

from ma_sh.Model.mash import Mash
from ma_sh.Module.o3d_viewer import O3DViewer


def demo():
    if False:
        mash_params_folder_path = "./output/dataset/"

        class_name_list = os.listdir(mash_params_folder_path)

        for class_name in class_name_list:
            mash_params_file_path = (
                mash_params_folder_path
                + class_name
                + "/models/model_normalized_obj.npy"
            )

            if not os.path.exists(mash_params_file_path):
                continue

            mash = Mash.fromParamsFile(mash_params_file_path, device="cpu")

            mash.renderSamplePoints()

    if False:
        mash_params_file_path = "./output/mac_chair_2.npy"

        mash = Mash.fromParamsFile(mash_params_file_path, 18, 4000, 0.4, device="cpu")

        mash.renderSamplePoints()

    if True:
        o3d_viewer = O3DViewer()
        o3d_viewer.createWindow()

        mash_folder_path = "./output/20240421_00:51:09/"

        mash_filename_list = os.listdir(mash_folder_path)
        mash_filename_list.sort()

        print(mash_filename_list)
        exit()

        mash_params_file_path = "./output/mac_chair_2.npy"

        mash = Mash.fromParamsFile(mash_params_file_path, 18, 4000, 0.4, device="cpu")

        mash.renderSamplePoints()
    return True
