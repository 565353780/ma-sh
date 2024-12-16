import os
import numpy as np

from open_clip_detect.Module.detector import Detector as CLIPDetector
from dino_v2_detect.Module.detector import Detector as DINODetector
from ulip_manage.Module.detector import Detector as ULIPDetector

from ma_sh.Module.Convertor.base_convertor import BaseConvertor


class Convertor(BaseConvertor):
    def __init__(
        self,
        source_root_folder_path: str,
        target_root_folder_path: str,
        model_file_path: str,
        mode: str,
        device: str = "cuda:0",
    ) -> None:
        super().__init__(source_root_folder_path, target_root_folder_path)

        self.model_file_path = model_file_path
        self.mode = mode
        self.device = device

        assert mode in ['clip', 'dino', 'ulip']

        if self.mode == 'clip':
            self.clip_detector = CLIPDetector(model_file_path, self.device, False)
        if self.mode == 'dino':
            self.dino_detector = DINODetector(model_file_path, self.device)
        elif self.mode == 'ulip':
            open_clip_model_file_path = '/home/chli/Model/CLIP-ViT-bigG-14-laion2B-39B-b160k/open_clip_pytorch_model.bin'
            self.ulip_detector = ULIPDetector(model_file_path, open_clip_model_file_path, device)
        return

    def convertData(self, source_path: str, target_path: str) -> bool:
        image_filename_list = os.listdir(source_path)

        #FIXME: check if all images are captured
        if len(image_filename_list) < 12:
            return False

        image_embedding_dict = {}

        for image_filename in image_filename_list:
            if ".png" not in image_filename:
                continue

            image_file_path = source_path + image_filename

            if self.mode == 'clip':
                image_embedding = self.clip_detector.detectImageFile(image_file_path).cpu().numpy()
            elif self.mode == 'dino':
                image_embedding = self.dino_detector.detectFile(image_file_path).cpu().numpy()
            elif self.mode == 'ulip':
                image_embedding = self.ulip_detector.encodeImageFile(image_file_path).unsqueeze(0).cpu().numpy()
            else:
                print('[ERROR][Convertor::convertData]')
                print('\t mode not valid!')
                print('\t mode:', self.mode)
                return False

            image_embedding_dict[image_filename] = image_embedding

        if len(image_embedding_dict.keys()) == 0:
            print('[ERROR][Convertor::convertData]')
            print("\t no valid image found!")
            print("\t source_path:", source_path)
            return False

        np.save(target_path, image_embedding_dict)

        return True
