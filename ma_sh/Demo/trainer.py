import torch

from ma_sh.Config.custom_path import mesh_file_path_dict
from ma_sh.Module.trainer import Trainer


def demo():
    anchor_num = 100
    mask_degree_max = 1
    sh_degree_max = 3
    mask_boundary_sample_num = 10
    sample_point_scale = 0.8
    delta_theta_angle = 1.0
    use_inv = True
    idx_dtype = torch.int64
    dtype = torch.float64
    device = "cuda:0"

    epoch = 10000
    lr = 1e-1
    weight_decay = 1e-4
    factor = 0.99
    patience = 1
    min_lr = 1e-4

    render = False

    mesh_name = "linux_2"

    save_result_folder_path = "auto"
    save_log_folder_path = "auto"

    mesh_file_path = mesh_file_path_dict[mesh_name]

    gt_points_num = 10000

    save_params_file_path = "./output/" + mesh_name + ".npy"
    save_pcd_file_path = "./output/" + mesh_name + ".ply"
    overwrite = True
    print_progress = True

    trainer = Trainer(
        anchor_num,
        mask_degree_max,
        sh_degree_max,
        mask_boundary_sample_num,
        sample_point_scale,
        delta_theta_angle,
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
        save_result_folder_path,
        save_log_folder_path,
    )

    trainer.loadMeshFile(mesh_file_path)
    trainer.autoTrainMash(gt_points_num)
    trainer.mash.saveParamsFile(save_params_file_path, overwrite)
    trainer.mash.saveAsPcdFile(save_pcd_file_path, overwrite, print_progress)

    # trainer.o3d_viewer.run()
    # trainer.mash.renderSamplePoints()
    return True
