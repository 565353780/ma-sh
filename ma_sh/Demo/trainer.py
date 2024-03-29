import torch

from ma_sh.Config.custom_path import mesh_file_path_dict
from ma_sh.Module.trainer import Trainer


def demo():
    anchor_num = 20
    mask_degree_max = 4
    sh_degree_max = 4
    mask_boundary_sample_num = 10
    sample_polar_num = 4000
    sample_point_scale = 0.5
    use_inv = True
    idx_dtype = torch.int64
    dtype = torch.float64
    device = "cpu"

    warm_epoch_step_num = 10
    warm_epoch_num = 2
    finetune_step_num = 10000
    lr = 1e-2
    weight_decay = 1e-4
    factor = 0.98
    patience = 1
    min_lr = 1e-4

    render = True
    render_init_only = False

    mesh_name = "mac_airplane"

    save_result_folder_path = "auto"
    save_log_folder_path = "auto"

    mesh_file_path = mesh_file_path_dict[mesh_name]

    gt_points_num = 4000

    save_params_file_path = "./output/" + mesh_name + ".npy"
    save_pcd_file_path = "./output/" + mesh_name + ".ply"
    overwrite = True
    print_progress = True

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
        warm_epoch_step_num,
        warm_epoch_num,
        finetune_step_num,
        lr,
        weight_decay,
        factor,
        patience,
        min_lr,
        render,
        save_result_folder_path,
        save_log_folder_path,
        render_init_only,
    )

    trainer.loadMeshFile(mesh_file_path)
    trainer.autoTrainMash(gt_points_num)
    trainer.mash.saveParamsFile(save_params_file_path, overwrite)
    trainer.mash.saveAsPcdFile(save_pcd_file_path, overwrite, print_progress)

    trainer.o3d_viewer.run()
    trainer.mash.renderSamplePoints()
    return True
