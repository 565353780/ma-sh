import sys
sys.path.append('../open-clip-detect/')
sys.path.append('../dino-v2-detect/')
# sys.path.append('../ulip-manage/')

from ma_sh.Config.custom_path import toDatasetRootPath, toModelRootPath
from ma_sh.Module.Convertor.encode_image import Convertor


def demo():
    dataset_root_folder_path = toDatasetRootPath()
    assert dataset_root_folder_path is not None

    model_root_folder_path = toModelRootPath()
    assert model_root_folder_path is not None

    model_file_path_dict = {
        'clip': model_root_folder_path + 'open_clip/DFN5B-CLIP-ViT-H-14-378.bin',
        'dino_s': model_root_folder_path + 'DINOv2/dinov2_vits14_reg4_pretrain.pth',
        'dino_b': model_root_folder_path + 'DINOv2/dinov2_vitb14_reg4_pretrain.pth',
        'dino_l': model_root_folder_path + 'DINOv2/dinov2_vitl14_reg4_pretrain.pth',
        'dino_g': model_root_folder_path + 'DINOv2/dinov2_vitg14_reg4_pretrain.pth',
        'ulip': model_root_folder_path + 'ULIP2/pretrained_models_ckpt_zero-sho_classification_pointbert_ULIP-2.pt',
    }
    mode = 'dino_s'
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
