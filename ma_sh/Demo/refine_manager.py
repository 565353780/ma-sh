import torch

from ma_sh.Module.refine_manager import RefineManager


def demoFile():
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

    mash_file_path = "/home/chli/github/ASDF/conditional-flow-matching/output/sample/20241202_18:30:59/iter-9/category/03001627/mash/sample_2.npy"
    save_mash_file_path = "./output/mash.npy"
    save_pcd_file_path = "./output/pcd.ply"
    overwrite = True
    print_progress = True

    refine_manager = RefineManager(
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
        patience)

    refine_manager.refineFile(
        mash_file_path,
        save_mash_file_path,
        save_pcd_file_path,
        overwrite,
        print_progress)

    return True

def demoFolder():
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

    mash_folder_path = "/home/chli/github/ASDF/conditional-flow-matching/output/sample/20241218_15:08:27/"
    save_mash_folder_path = "/home/chli/github/ASDF/conditional-flow-matching/output/sample/20241218_15:08:27/refine_mash/"
    save_pcd_folder_path = "/home/chli/github/ASDF/conditional-flow-matching/output/sample/20241218_15:08:27/refine_pcd/"
    overwrite = True

    refine_manager = RefineManager(
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
        patience)

    refine_manager.refineFolder(
        mash_folder_path,
        save_mash_folder_path,
        save_pcd_folder_path,
        overwrite)

    return True
