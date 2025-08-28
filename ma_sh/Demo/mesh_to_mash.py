import sys

sys.path.append("../chamfer-distance")
sys.path.append("../data-convert")
sys.path.append("../diff-curvature")
sys.path.append("../mesh-graph-cut")

import torch
from typing import Union

from ma_sh.Module.Convertor.mesh_to_mash import Convertor as MeshToMashConvertor


def demo_mesh_to_mash(
    data_space: str,
    output_space: str,
    rel_data_path: str,
    anchor_num: int = 8192,
    mask_degree_max: int = 2,
    sh_degree_max: int = 2,
    dtype=torch.float64,
    device: str = "cuda",
    sample_phi_num: int = 40,
    sample_theta_num: int = 40,
    dist_max: float = 1.0 / 200,
    points_per_submesh: int = 1024,
    lr: float = 4.0,
    min_lr: float = 1e-1,
    warmup_step_num: int = 80,
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

    mesh_to_mash_convertor = MeshToMashConvertor(
        data_space,
        output_space + "mash/",
        anchor_num=anchor_num,
        mask_degree_max=mask_degree_max,
        sh_degree_max=sh_degree_max,
        dtype=dtype,
        device=device,
        sample_phi_num=sample_phi_num,
        sample_theta_num=sample_theta_num,
        dist_max=dist_max,
        points_per_submesh=points_per_submesh,
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

    mesh_to_mash_convertor.convertOneShape(rel_base_path, data_type, ".npy")

    # mesh_to_mash_convertor.convertAll(mash_data_type_list)
    return True
