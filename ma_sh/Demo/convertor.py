import os
import torch

from ma_sh.Module.convertor import Convertor


def demo():
    HOME = os.environ["HOME"]

    shape_root_folder_path = HOME + "/chLi/Dataset/ShapeNet/Core/ShapeNetCore.v2/"
    save_root_folder_path = HOME + "/chLi/Dataset/Mash/ShapeNet/"
    if False:
        shape_root_folder_path = HOME + "/Dataset/aro_net/data/shapenet/00_meshes/"
        save_root_folder_path = HOME + "/Dataset/aro_net/data/shapenet/"

    force_start = False
    gt_points_num = 10000
    anchor_num = 40
    mask_degree_max = 4
    sh_degree_max = 4
    mask_boundary_sample_num = 18
    sample_polar_num = 4000
    sample_point_scale = 0.4
    use_inv = True
    idx_dtype = torch.int64
    dtype = torch.float64
    device = "cuda:0"

    warm_epoch_step_num = 20
    warm_epoch_num = 10
    finetune_step_num = 400
    lr = 1e-2
    weight_decay = 1e-4
    factor = 0.9
    patience = 1
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
