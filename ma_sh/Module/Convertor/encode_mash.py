import os
import numpy as np

from ma_sh.Method.data import toNumpy
from ma_sh.Method.path import createFileFolder

from mash_autoencoder.Module.detector import Detector


class Convertor(object):
    def __init__(
        self,
        dataset_root_folder_path: str,
        model_file_path: str,
        device: str = "cuda:0",
        force_start: bool = False,
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path
        self.device = device
        self.force_start = force_start

        self.mash_folder_path = (
            self.dataset_root_folder_path + "MashV3/"
        )
        self.encoded_mash_folder_path = (
            self.dataset_root_folder_path + "EncodedMash/"
        )
        self.tag_folder_path = self.dataset_root_folder_path + "Tag/EncodedMash/"

        self.detector = Detector(model_file_path, device=self.device)
        return

    def convertOneShape(
        self, dataset_name: str, class_name: str, model_id: str
    ) -> bool:
        rel_file_path = dataset_name + "/" + class_name + "/" + model_id

        mash_file_path = (
            self.mash_folder_path + rel_file_path + ".npy"
        )

        if not os.path.exists(mash_file_path):
            print("[ERROR][Convertor::convertOneShape]")
            print("\t shape folder not exist!")
            print("\t mash_file_path:", mash_file_path)
            return False

        finish_tag_file_path = self.tag_folder_path + rel_file_path + "/finish.txt"

        if os.path.exists(finish_tag_file_path):
            return True

        start_tag_file_path = self.tag_folder_path + rel_file_path + "/start.txt"

        if os.path.exists(start_tag_file_path):
            if not self.force_start:
                return True

        createFileFolder(start_tag_file_path)

        with open(start_tag_file_path, "w") as f:
            f.write("\n")

        encoded_mash_file_path = (
            self.encoded_mash_folder_path + rel_file_path + ".npy"
        )

        createFileFolder(encoded_mash_file_path)

        results = self.detector.encodeFile(mash_file_path)

        if results is None:
            print("[ERROR][Convertor::convertOneShape]")
            print("\t detectFile failed!")
            print("\t mash_file_path:", mash_file_path)
            return False

        encoded_mash = toNumpy(results['x'][0])

        np.save(encoded_mash_file_path, encoded_mash)

        with open(finish_tag_file_path, "w") as f:
            f.write("\n")

        return True

    def convertAll(self) -> bool:
        print("[INFO][Convertor::convertAll]")
        print("\t start convert all shapes to mashes...")
        solved_shape_num = 0

        dataset_folder_path = self.mash_folder_path + "ShapeNet/"

        classname_list = os.listdir(dataset_folder_path)
        classname_list.sort()
        for classname in classname_list:
            class_folder_path = dataset_folder_path + classname + "/"

            modelid_list = os.listdir(class_folder_path)
            modelid_list.sort()

            for model_file_name in modelid_list:
                modelid = model_file_name.split(".obj")[0]

                self.convertOneShape("ShapeNet", classname, modelid)

                solved_shape_num += 1
                print("solved shape num:", solved_shape_num)
        return True
