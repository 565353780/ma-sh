import torch

from ma_sh.Module.trainer import Trainer
from ma_sh.Module.timer import Timer


def demo(
    gt_points_file_path: str,
    anchor_num: int = 400,
    save_freq: int = 1,
    save_log_folder_path: str = 'auto',
    save_result_folder_path: str = 'auto',
):
    #anchor_num = 400
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
    warmup_step_num = 80
    warmup_epoch = 4
    factor = 0.8
    patience = 2

    render = False
    render_freq = 1
    render_init_only = False
    # save_freq = 1

    gt_points_num = 400000

    # save_result_folder_path = None
    # save_log_folder_path = None

    # save_result_folder_path = 'auto'
    # save_log_folder_path = 'auto'

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

    if False:
        trainer.loadMeshFile(mesh_file_path)
    else:
        '''
        mesh_id = 0
        mesh_name_list = [
            '03001627/1016f4debe988507589aae130c1f06fb',
            '02691156/1066b65c30d153e04c3a35cee92bb95b',
            "04090263/22d2782aa73ea40960abd8a115f9899",
            "03001627/46e1939ce6ee14d6a4689f3cf5c22e6",
            "03001627/1b8e84935fdc3ec82be289de70e8db31",
            "03001627/e71d05f223d527a5f91663a74ccd2338",
        ]
        gt_points_file_path = '/home/chli/chLi2/Dataset/SampledPcd_Manifold/ShapeNet/' + mesh_name_list[mesh_id] + '.npy'
        '''

        trainer.loadGTPointsFile(gt_points_file_path, gt_points_num)

    timer = Timer()
    trainer.autoTrainMash(gt_points_num)

    print('finish training, spend time :', timer.now())
    return True
