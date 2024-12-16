import sys
sys.path.append('../open-clip-detect/')
sys.path.append('../dino-v2-detect/')
sys.path.append('../ulip-manage/')

import os

from ma_sh.Config.custom_path import toDatasetRootPath
from ma_sh.Module.Convertor.encode_image import Convertor


def demo():
    dataset_root_folder_path = toDatasetRootPath()
    if dataset_root_folder_path is None:
        print('[ERROR][encode_image::demo]')
        print('\t toDatasetRootPath failed!')
        return False

    model_file_path_dict = {
        'clip': os.environ['HOME'] + '/Model/open_clip/DFN5B-CLIP-ViT-H-14-378.bin',
        'dino': os.environ['HOME'] + '/Model/DINOv2/dinov2_vitg14_reg4_pretrain.pth',
        'ulip': os.environ['HOME'] + '/chLi/Model/ULIP2/pretrained_models_ckpt_zero-sho_classification_pointbert_ULIP-2.pt',
    }
    mode = 'dino'
    device = "cuda:0"
    force_start = False

    convertor = Convertor(
        dataset_root_folder_path,
        model_file_path_dict[mode],
        mode,
        device,
        force_start,
    )

    convertor.convertAll()
    return True
