import sys
sys.path.append('../open-clip-detect/')
sys.path.append('../dino-v2-detect/')
sys.path.append('../ulip-manage/')

from ma_sh.Config.custom_path import toDatasetRootPath
from ma_sh.Module.Convertor.encode_image import Convertor


def demo():
    dataset_root_folder_path = toDatasetRootPath()
    if dataset_root_folder_path is None:
        print('[ERROR][encode_image::demo]')
        print('\t toDatasetRootPath failed!')
        return False

    model_file_path_dict = {
        'clip': dataset_root_folder_path + '../Model/open_clip/DFN5B-CLIP-ViT-H-14-378.bin',
        'dino': dataset_root_folder_path + '../Model/DINOv2/dinov2_vitl14_reg4_pretrain.pth',
        'ulip': dataset_root_folder_path + '../Model/ULIP2/pretrained_models_ckpt_zero-sho_classification_pointbert_ULIP-2.pt',
    }
    mode = 'dino'
    device = "cuda:0"
    source_data_type = '.png'
    target_data_type = '.npy'

    convertor = Convertor(
        dataset_root_folder_path + "Objaverse_82K/render/",
        dataset_root_folder_path + "Objaverse_82K/render_" + mode + "/",
        model_file_path_dict[mode],
        mode,
        device,
    )

    convertor.convertAll(source_data_type, target_data_type)
    return True
