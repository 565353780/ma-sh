import os
import torch

from ma_sh.Demo.mash_to_pcd import demo_mash_to_pcd


if __name__ == "__main__":
    data_space = os.environ["HOME"] + "/chLi/Results/ma-sh/MeshTrainer/results/"
    output_space = os.environ["HOME"] + "/chLi/Results/ma-sh/MeshTrainer/pcd/"
    cuda_id = "0"

    sample_phi_num = 10
    sample_theta_num = 10
    dtype = torch.float32
    device = "cuda"

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id
    demo_mash_to_pcd(
        data_space,
        output_space,
        sample_phi_num=sample_phi_num,
        sample_theta_num=sample_theta_num,
        dtype=dtype,
        device=device,
    )
