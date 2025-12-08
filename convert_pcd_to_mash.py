import os
import torch

from ma_sh.Demo.pcd_to_mash import demo_pcd_to_mash


if __name__ == "__main__":
    """
    data_space: the root folder of the shape dataset
    output_space: the root folder of the generated results
    rel_data_path: data_file_path = data_space + rel_data_path
    """
    home = os.environ["HOME"]
    data_space = home + "/chLi/Dataset/Bunny/bunny/pcd/"
    output_space = home + "/chLi/Dataset/Bunny/bunny/"
    rel_data_path = "bun_zipper.ply"
    cuda_id = "0"

    gt_points_num = 400000
    anchor_num = 50
    mask_degree_max = 3
    sh_degree_max = 2
    use_inv = True
    dtype = torch.float32
    device = "cuda"
    mask_boundary_sample_num = 90
    sample_point_num = 1000
    sample_point_scale = 0.8
    lr = 2e-3
    min_lr = 1e-3
    warmup_step_num = 80
    warmup_epoch = 4
    factor = 0.8
    patience = 2
    render = False
    render_freq = 1
    render_init_only = False
    save_freq = -1
    save_result_folder_path = None
    save_log_folder_path = None

    """
    anchor_num = 2000
    save_freq = 1
    save_result_folder_path = home + "/chLi/Results/ma-sh/MeshTrainer/results/bunny-v2/"
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id
    demo_pcd_to_mash(
        data_space,
        output_space,
        rel_data_path,
        gt_points_num=gt_points_num,
        anchor_num=anchor_num,
        mask_degree_max=mask_degree_max,
        sh_degree_max=sh_degree_max,
        use_inv=use_inv,
        dtype=dtype,
        device=device,
        mask_boundary_sample_num=mask_boundary_sample_num,
        sample_point_num=sample_point_num,
        sample_point_scale=sample_point_scale,
        lr=lr,
        min_lr=min_lr,
        warmup_step_num=warmup_step_num,
        warmup_epoch=warmup_epoch,
        factor=factor,
        patience=patience,
        render=render,
        render_freq=render_freq,
        render_init_only=render_init_only,
        save_freq=save_freq,
        save_result_folder_path=save_result_folder_path,
        save_log_folder_path=save_log_folder_path,
    )
