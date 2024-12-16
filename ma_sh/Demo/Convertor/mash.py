import torch

from ma_sh.Config.custom_path import toDatasetRootPath
from ma_sh.Module.Convertor.mash import Convertor


def demo(device: str = 'cuda:0'):
    dataset_root_folder_path = toDatasetRootPath()

    if dataset_root_folder_path is None:
        print('[ERROR][sample_pcd_objaverse::demo]')
        print('\t dataset not found!')
        return False

    gt_points_num = 400000
    anchor_num = 400
    mask_degree_max = 3
    sh_degree_max = 2
    mask_boundary_sample_num = 90
    sample_polar_num = 1000
    sample_point_scale = 0.8
    use_inv = True
    idx_dtype = torch.int64
    dtype = torch.float32
    # device = "cuda:0"

    lr = 2e-3
    min_lr = 1e-3
    warmup_step_num = 80
    warmup_epoch = 4
    factor = 0.8
    patience = 2

    convertor = Convertor(
        dataset_root_folder_path,
        gt_points_num,
        anchor_num,
        mask_degree_max,
        sh_degree_max,
        mask_boundary_sample_num,
        sample_polar_num,
        sample_point_scale,
        use_inv,
        idx_dtype,
        dtype,
        device,
        lr,
        min_lr,
        warmup_step_num,
        warmup_epoch,
        factor,
        patience,
    )

    convertor.convertAll()
    return True
