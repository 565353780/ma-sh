import sys

sys.path.append("../chamfer-distance")
sys.path.append("../data-convert")

import torch
from typing import Union

from ma_sh.Module.Convertor.pcd_to_mash import Convertor as PcdToMashConvertor


def demo_pcd_to_mash(
    data_space: str,
    output_space: str,
    rel_data_path: str,
    gt_points_num: int = 400000,
    anchor_num: int = 1600,
    mask_degree_max: int = 3,
    sh_degree_max: int = 2,
    use_inv: bool = True,
    dtype=torch.float64,
    device: str = "cuda",
    mask_boundary_sample_num: int = 90,
    sample_point_num: int = 1000,
    sample_point_scale: float = 0.8,
    lr: float = 2e-3,
    min_lr: float = 1e-3,
    warmup_step_num: int = 80,
    warmup_epoch: int = 4,
    factor: float = 0.8,
    patience: int = 2,
    render: bool = False,
    render_freq: int = 1,
    render_init_only: bool = False,
    save_freq: int = -1,
    save_result_folder_path: Union[str, None] = None,
    save_log_folder_path: Union[str, None] = None,
):
    data_type = "." + rel_data_path.split(".")[-1]
    rel_base_path = rel_data_path[: -len(data_type)]

    pcd_to_mash_convertor = PcdToMashConvertor(
        data_space,
        output_space + "mash/",
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

    pcd_to_mash_convertor.convertOneShape(rel_base_path, data_type, ".npy")

    # pcd_to_mash_convertor.convertAll(mash_data_type_list)
    return True
