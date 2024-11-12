import os
import numpy as np

from ma_sh.Method.path import createFileFolder

from shapeglot_dataset_manage.Module.text_loader import TextLoader
from ulip_manage.Module.detector import Detector

class Convertor(object):
    def __init__(
        self,
        dataset_root_folder_path: str,
        device: str = "cuda:0",
        force_start: bool = False,
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path
        self.device = device
        self.force_start = force_start

        self.mash_folder_path = (
            self.dataset_root_folder_path + "MashV4/"
        )
        self.text_embedding_folder_path = (
            self.dataset_root_folder_path + "TextEmbedding_ShapeGlot/"
        )
        self.tag_folder_path = self.dataset_root_folder_path + "Tag/TextEmbedding_ShapeGlot/"

        model_file_path = '/home/chli/chLi/Model/ULIP2/pretrained_models_ckpt_zero-sho_classification_pointbert_ULIP-2.pt'
        open_clip_model_file_path = '/home/chli/Model/CLIP-ViT-bigG-14-laion2B-39B-b160k/open_clip_pytorch_model.bin'
        self.detector = Detector(model_file_path, open_clip_model_file_path, device)

        dataset_root_folder_path = '/home/chli/chLi/Dataset/ShapeGlot/data/'
        text_loader = TextLoader(dataset_root_folder_path)
        text_loader.loadTexts()
        self.text_dict = text_loader.text_dict
        return

    def convertOneShape(
        self, dataset_name: str, class_name: str, model_id: str
    ) -> bool:
        rel_file_path = dataset_name + "/" + class_name + "/" + model_id

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

        if class_name not in self.text_dict.keys():
            return True

        if model_id not in self.text_dict[class_name].keys():
            return True

        text_list = self.text_dict[class_name][model_id]

        text_embedding_file_path = (
            self.text_embedding_folder_path + rel_file_path + ".npy"
        )

        createFileFolder(text_embedding_file_path)

        text_embedding_dict = {}

        for i, text in enumerate(text_list):
            text_embedding = self.detector.encodeText(text).unsqueeze(0).cpu().numpy()

            text_embedding_dict[str(i)] = text_embedding

        if len(text_embedding_dict.keys()) == 0:
            print("[WARN][Convertor::convertOneShape]")
            print("\t no valid text found!")
            return True

        np.save(text_embedding_file_path, text_embedding_dict)

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
            if classname not in self.text_dict.keys():
                continue

            class_folder_path = dataset_folder_path + classname + "/"

            modelid_list = os.listdir(class_folder_path)
            modelid_list.sort()

            for model_file_name in modelid_list:
                if solved_shape_num < 0:
                    solved_shape_num += 1
                    continue

                modelid = model_file_name.split(".npy")[0]

                self.convertOneShape("ShapeNet", classname, modelid)

                solved_shape_num += 1
                print("solved shape num:", solved_shape_num)
        return True
