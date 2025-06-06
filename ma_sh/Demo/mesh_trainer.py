import sys

sys.path.append("../wn-nc/")
sys.path.append("../chamfer-distance/")
sys.path.append("../diff-curvature")
sys.path.append("../mesh-graph-cut")

import torch
from typing import Union

from ma_sh.Module.mesh_trainer import MeshTrainer
from ma_sh.Module.timer import Timer


def demo(
    mesh_file_path: str,
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
    sample_phi_num = 40
    sample_theta_num = 40
    use_inv = True
    idx_dtype = torch.int64
    dtype = torch.float32
    device = "cuda:7"

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

    trainer = MeshTrainer(
        anchor_num,
        mask_degree_max,
        sh_degree_max,
        sample_phi_num,
        sample_theta_num,
        use_inv,
        idx_dtype,
        dtype,
        device,
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

    print("start load GT mesh...")
    trainer.loadMeshFile(mesh_file_path)

    timer = Timer()
    print("start optimizing MASH...")
    trainer.autoTrainMash()

    print("finish training, spend time :", timer.now())
    return True
