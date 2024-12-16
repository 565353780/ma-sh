import os
import numpy as np

from open_clip_detect.Module.detector import Detector as CLIPDetector
from dino_v2_detect.Module.detector import Detector as DINODetector
from ulip_manage.Module.detector import Detector as ULIPDetector

from ma_sh.Method.path import createFileFolder


class Convertor(object):
    def __init__(
        self,
        dataset_root_folder_path: str,
        model_file_path: str,
        mode: str,
        device: str = "cuda:0",
        force_start: bool = False,
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path
        self.model_file_path = model_file_path

        self.mode = mode
        assert mode in ['clip', 'dino', 'ulip']


        self.device = device
        self.force_start = force_start

        self.captured_image_folder_path = (
            self.dataset_root_folder_path + "Objaverse_82K/render/"
        )
        self.image_embedding_folder_path = (
            self.dataset_root_folder_path + "Objaverse_82K/render_" + mode + "/"
        )
        self.tag_folder_path = self.dataset_root_folder_path + "Tag/Objaverse_82K_render_" + mode + "/"

        local_path = '/home/chli/Dataset/'
        if os.path.exists(local_path):
            self.image_embedding_folder_path = (
                local_path + "Objaverse_82K/render_" + mode + "/"
            )
            self.tag_folder_path = local_path + "Tag/Objaverse_82K_render_" + mode + "/"

        if self.mode == 'clip':
            self.clip_detector = CLIPDetector(model_file_path, self.device, False)
        if self.mode == 'dino':
            self.dino_detector = DINODetector(model_file_path, self.device)
        elif self.mode == 'ulip':
            open_clip_model_file_path = '/home/chli/Model/CLIP-ViT-bigG-14-laion2B-39B-b160k/open_clip_pytorch_model.bin'
            self.ulip_detector = ULIPDetector(model_file_path, open_clip_model_file_path, device)
        return

    def convertOneShape(
        self, model_id: str
    ) -> bool:
        rel_file_path = model_id

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
            '''
            #FIXME: check if all images are embedded
            image_embedding_file_path = (
                self.image_embedding_folder_path + rel_file_path + ".npy"
            )
            image_embedding_dict = np.load(image_embedding_file_path, allow_pickle=True).item()
            if len(list(image_embedding_dict.keys())) < 12:
                removeFile(finish_tag_file_path)
                removeFile(image_embedding_file_path)
            else:
                return True
            '''
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

        #FIXME: check if all images are captured
        if len(image_filename_list) < 12:
            return False

        image_embedding_dict = {}

        for image_filename in image_filename_list:
            if ".png" not in image_filename:
                continue

            image_file_path = captured_image_folder_path + image_filename

            if self.mode == 'clip':
                image_embedding = self.clip_detector.detectImageFile(image_file_path).cpu().numpy()
            elif self.mode == 'dino':
                image_embedding = self.dino_detector.detectFile(image_file_path).cpu().numpy()
            elif self.mode == 'ulip':
                image_embedding = self.ulip_detector.encodeImageFile(image_file_path).unsqueeze(0).cpu().numpy()
            else:
                print('[ERROR][Convertor::convertOneShape]')
                print('\t mode not valid!')
                print('\t mode:', self.mode)
                return False

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

        dataset_folder_path = self.captured_image_folder_path

        classname_list = os.listdir(dataset_folder_path)
        classname_list.sort()
        for classname in classname_list:
            class_folder_path = dataset_folder_path + classname + "/"

            modelid_list = os.listdir(class_folder_path)
            modelid_list.sort()

            for modelid in modelid_list:
                if solved_shape_num < 0:
                    solved_shape_num += 1
                    continue

                model_id = classname + '/' + modelid

                self.convertOneShape(model_id)

                solved_shape_num += 1
                print("solved shape num:", solved_shape_num)
        return True
