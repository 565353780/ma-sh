import torch

from ma_sh.Module.pcd_convertor import PcdConvertor


def demo():
    save_root_folder_path = "/home/chli/Dataset/Mash50/"
    save_root_folder_path = "/home/chli/Dataset/MashTest/100anc/"
    mask_boundary_sample_num = 36
    sample_polar_num = 20000
    sample_point_scale = 0.8
    use_inv = True
    idx_dtype = torch.int64
    dtype = torch.float64
    device = "cuda:0"

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
