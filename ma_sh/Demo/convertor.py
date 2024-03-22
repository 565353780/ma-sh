import torch

from ma_sh.Module.convertor import Convertor


def demo():
    shape_root_folder_path = "/home/chli/chLi/Dataset/ShapeNet/Core/ShapeNetCore.v2/"
    save_root_folder_path = "/home/chli/chLi/Dataset/Mash/ShapeNet/"
    shape_root_folder_path = (
        "/home/chli/Nutstore Files/paper-materials-ASDF/Dataset/mini/03001627/"
    )
    save_root_folder_path = "/home/chli/Dataset/MashTest/100anc/"
    force_start = False
    gt_points_num = 40000
    anchor_num = 100
    mask_degree_max = 1
    sh_degree_max = 3
    mask_boundary_sample_num = 36
    sample_polar_num = 40000
    sample_point_scale = 0.8
    use_inv = True
    idx_dtype = torch.int64
    dtype = torch.float64
    device = "cuda:0"

    epoch = 10000
    lr = 1e-2
    weight_decay = 1e-4
    factor = 0.99
    patience = 1
    min_lr = 1e-3

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
        epoch,
        lr,
        weight_decay,
        factor,
        patience,
        min_lr,
    )

    convertor.convertAll()
    return True
