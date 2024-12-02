import torch

from ma_sh.Module.refiner import Refiner


def demo():
    anchor_num = 400
    mask_degree_max = 3
    sh_degree_max = 2
    mask_boundary_sample_num = 90
    sample_polar_num = 1000
    sample_point_scale = 0.8
    use_inv = True
    idx_dtype = torch.int64
    dtype = torch.float32
    device = "cuda:0"

    lr = 2e-3
    min_lr = 1e-3
    warm_step_num = 40
    factor = 0.8
    patience = 2

    render = False
    render_freq = 1
    render_init_only = False

    save_result_folder_path = 'auto'
    save_log_folder_path = 'auto'

    mash_file_path = "/home/chli/github/ASDF/conditional-flow-matching/output/sample/20241202_18:30:59/iter-9/category/03001627/mash/sample_2.npy"
    save_params_file_path = "./output/mash.npy"
    save_pcd_file_path = "./output/pcd.ply"
    overwrite = True
    print_progress = True

    refiner = Refiner(
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
        lr,
        min_lr,
        warm_step_num,
        factor,
        patience,
        render,
        render_freq,
        render_init_only,
        save_result_folder_path,
        save_log_folder_path,
    )

    refiner.loadParamsFile(mash_file_path)
    refiner.autoTrainMash()
    refiner.mash.saveParamsFile(save_params_file_path, overwrite)
    refiner.mash.saveAsPcdFile(save_pcd_file_path, overwrite, print_progress)

    if refiner.o3d_viewer is not None:
        refiner.o3d_viewer.run()

    # refiner.mash.renderSamplePoints()

    return True
