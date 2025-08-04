import sys

sys.path.append("../voxel-converter")
sys.path.append("../chamfer-distance")
sys.path.append("../data-convert")
sys.path.append("../diff-curvature")
sys.path.append("../mesh-graph-cut")

import os
import torch

from ma_sh.Demo.npz_to_mash import demo_npz_to_mash


if __name__ == "__main__":
    """
    data_space: the root folder of the shape dataset
    output_space: the root folder of the generated results
    rel_data_path: data_file_path = data_space + rel_data_path
    """
    data_space = os.environ["HOME"] + "/chLi/Dataset/ShuMei/data/"
    output_space = os.environ["HOME"] + "/chLi/Dataset/ShuMei/"
    rel_data_path = "0008dc75fb3648f2af4ca8c4d711e53e.npz"
    cuda_id = "0"

    anchor_num = 8192
    mask_degree_max = 2
    sh_degree_max = 2
    sample_phi_num = 40
    sample_theta_num = 40
    points_per_submesh = 1024
    dtype = torch.float32
    device = "cuda"
    lr = 4.0
    min_lr = 1e-1
    warmup_step_num = 80
    factor = 0.8
    patience = 2
    render = False
    render_freq = 1
    render_init_only = False
    save_freq = -1
    save_result_folder_path = None
    save_log_folder_path = None

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id
    demo_npz_to_mash(
        data_space,
        output_space,
        rel_data_path,
        anchor_num=anchor_num,
        mask_degree_max=mask_degree_max,
        sh_degree_max=sh_degree_max,
        sample_phi_num=sample_phi_num,
        sample_theta_num=sample_theta_num,
        points_per_submesh=points_per_submesh,
        dtype=dtype,
        device=device,
        lr=lr,
        min_lr=min_lr,
        warmup_step_num=warmup_step_num,
        factor=factor,
        patience=patience,
        render=render,
        render_freq=render_freq,
        render_init_only=render_init_only,
        save_freq=save_freq,
        save_result_folder_path=save_result_folder_path,
        save_log_folder_path=save_log_folder_path,
    )
