import torch
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

        dtype = torch.float16

        assert mode in ["clip", "dino_s", "dino_b", "dino_l", "dino_g", "ulip"]

        if self.mode == "clip":
            self.clip_detector = CLIPDetector(model_file_path, self.device, False)
        elif self.mode == "dino_s":
            self.dino_detector = DINODetector("small", model_file_path, dtype, self.device)
        elif self.mode == "dino_b":
            self.dino_detector = DINODetector("base", model_file_path, dtype, self.device)
        elif self.mode == "dino_l":
            self.dino_detector = DINODetector("large", model_file_path, dtype, self.device)
        elif self.mode == "dino_g":
            self.dino_detector = DINODetector("giant2", model_file_path, dtype, self.device)
        elif self.mode == "ulip":
            open_clip_model_file_path = "/home/chli/Model/CLIP-ViT-bigG-14-laion2B-39B-b160k/open_clip_pytorch_model.bin"
            self.ulip_detector = ULIPDetector(
                model_file_path, open_clip_model_file_path, device
            )
        return

    def convertData(self, source_path: str, target_path: str) -> bool:
        image_embedding_dict = {}

        try:
            if self.mode == "clip":
                image_embedding = (
                    self.clip_detector.detectImageFile(source_path).cpu().numpy()
                )
            elif 'dino' in self.mode:
                image_embedding = (
                    self.dino_detector.detectFile(source_path)
                )
                if self.dino_detector.dtype == torch.bfloat16:
                    image_embedding = image_embedding.view(torch.uint16).cpu().numpy()
                else:
                    image_embedding = image_embedding.cpu().numpy()
            elif self.mode == "ulip":
                image_embedding = (
                    self.ulip_detector.encodeImageFile(source_path)
                    .unsqueeze(0)
                    .cpu()
                    .numpy()
                )
            else:
                print("[ERROR][Convertor::convertData]")
                print("\t mode not valid!")
                print("\t mode:", self.mode)
                return False
        except KeyboardInterrupt:
            print('[INFO][Convertor::convertData]')
            print('\t program interrupted by the user (Ctrl+C).')
            exit()
        except:
            print("[ERROR][Convertor::convertData]")
            print("\t detectImageFile failed!")
            return False

        image_embedding_dict[self.mode] = image_embedding

        try:
            np.save(target_path, image_embedding_dict)
        except:
            print('[ERROR][Convertor::convertData]')
            print('\t np.save failed!')
            print('\t target_path:', target_path)
            return False

        return True
