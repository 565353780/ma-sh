import os
import torch

from ma_sh.Module.Convertor.mash import Convertor


def demo():
    HOME = os.environ["HOME"]

    shape_root_folder_path = HOME + "/chLi/Dataset/ShapeNet/Core/ShapeNetCore.v2/"
    shape_root_folder_path = HOME + "/chLi/Dataset/SDF/ShapeNet/manifold/"
    save_root_folder_path = HOME + "/chLi/Dataset/Mash/ShapeNet/"

    force_start = False
    gt_points_num = 400000
    anchor_num = 400
    mask_degree_max = 4
    sh_degree_max = 3
    mask_boundary_sample_num = 10
    sample_polar_num = 10000
    sample_point_scale = 0.4
    use_inv = True
    idx_dtype = torch.int64
    dtype = torch.float64
    device = "cuda:0"

    warm_epoch_step_num = 10
    warm_epoch_num = 40
    finetune_step_num = 2000
    lr = 5e-3
    weight_decay = 1e-10
    factor = 0.9
    patience = 4
    min_lr = 1e-4

    convertor = Convertor(
        shape_root_folder_path,
        save_root_folder_path,
        force_start,
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
        warm_epoch_step_num,
        warm_epoch_num,
        finetune_step_num,
        lr,
        weight_decay,
        factor,
        patience,
        min_lr,
    )

    convertor.convertAll()
    return True
