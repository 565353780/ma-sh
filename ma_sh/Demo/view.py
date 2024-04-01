import os
from ma_sh.Model.mash import Mash


def demo():
    mash_params_folder_path = "./output/dataset/"

    class_name_list = os.listdir(mash_params_folder_path)

    for class_name in class_name_list:
        mash_params_file_path = (
            mash_params_folder_path + class_name + "/models/model_normalized_obj.npy"
        )

        if not os.path.exists(mash_params_file_path):
            continue

        mash = Mash.fromParamsFile(mash_params_file_path, device="cpu")

        mash.renderSamplePoints()
    return True
