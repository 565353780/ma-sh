import os
import clip
import torch
import numpy as np
from PIL import Image

from ma_sh.Method.path import createFileFolder


class Convertor(object):
    def __init__(
        self,
        dataset_root_folder_path: str,
        clip_model_id: str = "ViT-L/14",
        device: str = "cuda:0",
        force_start: bool = False,
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path
        self.clip_model_id = clip_model_id
        self.device = device
        self.force_start = force_start

        self.captured_image_folder_path = (
            self.dataset_root_folder_path + "CapturedImage/"
        )
        self.image_embedding_folder_path = (
            self.dataset_root_folder_path + "ImageEmbedding/"
        )
        self.tag_folder_path = self.dataset_root_folder_path + "Tag/ImageEmbedding/"

        self.model, self.preprocess = clip.load(self.clip_model_id, device=self.device)
        self.model.eval()
        return

    def convertOneShape(
        self, dataset_name: str, class_name: str, model_id: str
    ) -> bool:
        rel_file_path = dataset_name + "/" + class_name + "/" + model_id

        captured_image_folder_path = (
            self.captured_image_folder_path + rel_file_path + "/"
        )

        if not os.path.exists(captured_image_folder_path):
            print("[ERROR][Convertor::convertOneShape]")
            print("\t shape folder not exist!")
            print("\t captured_image_folder_path:", captured_image_folder_path)
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

        image_embedding_file_path = (
            self.image_embedding_folder_path + rel_file_path + ".npy"
        )

        createFileFolder(image_embedding_file_path)

        image_filename_list = os.listdir(captured_image_folder_path)

        image_embedding_dict = {}

        for image_filename in image_filename_list:
            if ".png" not in image_filename:
                continue

            image_file_path = captured_image_folder_path + image_filename
            image = Image.open(image_file_path)
            image = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_embedding = (
                    self.model.encode_image(image).detach().clone().cpu().numpy()
                )

                image_embedding_dict[image_filename] = image_embedding

        if len(image_embedding_dict.keys()) == 0:
            print("[WARN][Convertor::convertOneShape]")
            print("\t no valid image found!")
            print("\t captured_image_folder_path:", captured_image_folder_path)
            return True

        np.save(image_embedding_file_path, image_embedding_dict)

        with open(finish_tag_file_path, "w") as f:
            f.write("\n")

        return True

    def convertAll(self) -> bool:
        print("[INFO][Convertor::convertAll]")
        print("\t start convert all shapes to mashes...")
        solved_shape_num = 0

        dataset_folder_path = self.captured_image_folder_path + "ShapeNet/"

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
