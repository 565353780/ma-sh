import sys

sys.path.append("../data-convert")

import torch

from ma_sh.Module.Convertor.mash_to_pcd import Convertor as MashToPcdConvertor


def demo_mash_to_pcd(
    data_space: str,
    output_space: str,
    sample_phi_num: int = 40,
    sample_theta_num: int = 40,
    dtype=torch.float32,
    device: str = "cuda",
):
    mash_to_pcd_convertor = MashToPcdConvertor(
        data_space,
        output_space,
        sample_phi_num=sample_phi_num,
        sample_theta_num=sample_theta_num,
        dtype=dtype,
        device=device,
    )

    mash_to_pcd_convertor.convertAll(".npy", "_pcd.ply", 1.0)
    return True
