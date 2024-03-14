import torch

from ma_sh.Config.custom_path import mesh_file_path_dict
from ma_sh.Module.trainer import Trainer


def demo():
    anchor_num = 40
    mask_degree_max = 5
    sh_degree_max = 3
    mask_boundary_sample_num = 100
    sample_polar_num = 100000
    idx_dtype = torch.int64
    dtype = torch.float64
    device = "cpu"

    epoch = 10000
    lr = 1e-2
    weight_decay = 1e-4
    factor = 0.8
    patience = 4
    min_lr = lr * 1e-1

    render = False

    mesh_name = "linux_bunny"

    save_folder_path = "./output/" + mesh_name + "/"
    direction_upscale = 2

    use_inv = False

    mesh_file_path = mesh_file_path_dict[mesh_name]

    gt_points_num = 20000

    save_params_file_path = "./output/" + mesh_name + ".npy"
    overwrite = True

    trainer = Trainer(
        anchor_num,
        mask_degree_max,
        sh_degree_max,
        mask_boundary_sample_num,
        sample_polar_num,
        idx_dtype,
        dtype,
        device,
        epoch,
        lr,
        weight_decay,
        factor,
        patience,
        min_lr,
        render,
        save_folder_path,
        direction_upscale,
    )

    trainer.loadMeshFile(mesh_file_path)
    trainer.autoTrainMash(gt_points_num)
    trainer.o3d_viewer.run()
    trainer.mash.renderSamplePoints()
    trainer.saveParams(save_params_file_path, overwrite)
    return True
