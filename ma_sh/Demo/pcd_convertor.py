import torch

from ma_sh.Module.pcd_convertor import PcdConvertor


def demo():
    save_root_folder_path = "/home/chli/Dataset/aro_net/data/shapenet/"
    mask_boundary_sample_num = 18
    sample_polar_num = 4000
    sample_point_scale = 0.4
    use_inv = True
    idx_dtype = torch.int64
    dtype = torch.float64
    device = "cpu"

    pcd_convertor = PcdConvertor(
        save_root_folder_path,
        mask_boundary_sample_num,
        sample_polar_num,
        sample_point_scale,
        use_inv,
        idx_dtype,
        dtype,
        device,
    )

    pcd_convertor.convertAll()
    return True
