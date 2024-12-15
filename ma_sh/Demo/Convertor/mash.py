import os
import torch

from ma_sh.Module.Convertor.mash import Convertor


def demo(device: str = 'cuda:0'):
    HOME = os.environ["HOME"]
    root_list = [
        '/mnt/data/jintian/chLi/Dataset/',
        HOME + '/chLi/Dataset/',
    ]
    dataset_root_folder_path = None
    for root in root_list:
        if os.path.exists(root):
            dataset_root_folder_path = root
            break

    if dataset_root_folder_path is None:
        print('[ERROR][sample_pcd_objaverse::demo]')
        print('\t dataset not found!')
        return False

    dataset_name = 'Objaverse_82K'

    gt_points_num = 400000
    anchor_num = 400
    mask_degree_max = 3
    sh_degree_max = 2
    mask_boundary_sample_num = 90
    sample_polar_num = 1000
    sample_point_scale = 0.8
    use_inv = True
    idx_dtype = torch.int64
    dtype = torch.float32
    # device = "cuda:0"

    lr = 2e-3
    min_lr = 1e-3
    warmup_step_num = 80
    warmup_epoch = 4
    factor = 0.8
    patience = 2

    force_start = False

    convertor = Convertor(
        dataset_root_folder_path,
        dataset_name,
        gt_points_num,
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
        warmup_step_num,
        warmup_epoch,
        factor,
        patience,
        force_start,
    )

    convertor.convertAll()
    return True
