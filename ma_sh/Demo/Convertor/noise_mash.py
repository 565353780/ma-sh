import os
import torch

from ma_sh.Module.Convertor.noise_mash import Convertor


def demo(gauss_sigma: float = 0.01):
    HOME = os.environ["HOME"]
    dataset_root_folder_path = HOME + "/chLi/Dataset/"

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
    device = "cuda:0"

    lr = 2e-3
    min_lr = 1e-3
    warmup_step_num = 80
    warmup_epoch = 4
    factor = 0.8
    patience = 2

    # gauss_sigma = 0.01

    force_start = False

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
        gauss_sigma,
        force_start,
    )

    convertor.convertAll()
    return True
