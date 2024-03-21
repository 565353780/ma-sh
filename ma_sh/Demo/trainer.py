import torch

from ma_sh.Config.custom_path import mesh_file_path_dict
from ma_sh.Module.trainer import Trainer


def demo():
    anchor_num = 100
    mask_degree_max = 1
    sh_degree_max = 3
    mask_boundary_sample_num = 36
    sample_polar_num = 1000
    sample_point_scale = 0.8
    use_inv = True
    idx_dtype = torch.int64
    dtype = torch.float64
    device = "cuda:0"

    epoch = 1000
    lr = 1e-1
    weight_decay = 1e-4
    factor = 0.9
    patience = 10
    min_lr = 1e-3

    render = True

    mesh_name = "linux_bunny"

    save_folder_path = "./output/" + mesh_name + "/"
    direction_upscale = 2

    mesh_file_path = mesh_file_path_dict[mesh_name]

    gt_points_num = 10000

    save_params_file_path = "./output/" + mesh_name + ".npy"
    overwrite = True

    trainer = Trainer(
        anchor_num,
        mask_degree_max,
        sh_degree_max,
        mask_boundary_sample_num,
        sample_polar_num,
        sample_point_scale,
        use_inv,
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
