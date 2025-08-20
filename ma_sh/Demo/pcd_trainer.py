import sys

sys.path.append("../chamfer-distance/")

import torch
from typing import Union

from ma_sh.Module.pcd_trainer import PcdTrainer
from ma_sh.Module.timer import Timer


def demo(
    gt_points_file_path: str,
    anchor_num: int = 400,
    mask_degree_max: int = 3,
    sh_degree_max: int = 2,
    save_freq: int = 1,
    save_log_folder_path: Union[str, None] = "auto",
    save_result_folder_path: Union[str, None] = "auto",
):
    # anchor_num = 400
    # mask_degree_max = 3
    # sh_degree_max = 2
    use_inv = True
    dtype = torch.float32
    device = "cuda:0"
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
    # save_freq = 1

    gt_points_num = 400000

    # save_result_folder_path = None
    # save_log_folder_path = None

    # save_result_folder_path = 'auto'
    # save_log_folder_path = 'auto'

    pcd_trainer = PcdTrainer(
        anchor_num,
        mask_degree_max,
        sh_degree_max,
        use_inv,
        dtype,
        device,
        mask_boundary_sample_num,
        sample_point_num,
        sample_point_scale,
        lr,
        min_lr,
        warmup_step_num,
        warmup_epoch,
        factor,
        patience,
        render,
        render_freq,
        render_init_only,
        save_freq,
        save_result_folder_path,
        save_log_folder_path,
    )

    print("start load GT data...")
    pcd_trainer.loadGTPointsFile(gt_points_file_path, gt_points_num)

    timer = Timer()
    print("start optimizing MASH...")
    pcd_trainer.autoTrainMash(gt_points_num)

    print("finish training, spend time :", timer.now())
    return True
