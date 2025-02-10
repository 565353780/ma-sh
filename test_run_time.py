import torch

from ma_sh.Module.trainer import Trainer
from ma_sh.Module.timer import Timer


def testSpeed(
    gt_mesh_file_path: str,
    gt_points_num: int,
    anchor_num: int,
    sample_point_num: int,
):
    #anchor_num = 400
    mask_degree_max = 3
    sh_degree_max = 2
    mask_boundary_sample_num = 90
    # sample_point_num = 1000
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
    save_freq = -1

    # gt_points_num = 400000

    save_result_folder_path = None
    save_log_folder_path = None

    # save_result_folder_path = 'auto'
    # save_log_folder_path = 'auto'

    trainer = Trainer(
        anchor_num,
        mask_degree_max,
        sh_degree_max,
        mask_boundary_sample_num,
        sample_point_num,
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

    if gt_mesh_file_path.endswith('.npy'):
        trainer.loadGTPointsFile(gt_mesh_file_path, gt_points_num)
    else:
        trainer.loadMeshFile(gt_mesh_file_path)

    timer = Timer()
    trainer.autoTrainMash(gt_points_num)

    print('finish training, spend time :', timer.now())
    return True

def testSpeedForSampleNum():
    gt_mesh_file_path = '/home/chli/chLi/Dataset/Famous/bunny.ply'
    gt_points_num_list = [1000, 2000, 5000, 10000, 20000, 40000]
    sample_point_num_list = [int(i / 40) for i in gt_points_num_list]

    for i in range(len(gt_points_num_list)):
        gt_points_num = gt_points_num_list[i]
        sample_point_num = sample_point_num_list[i]
        print('start testSpeed on gt_points_num:', gt_points_num, 'sample_point_num:', sample_point_num)
        testSpeed(
            gt_mesh_file_path,
            gt_points_num,
            400,
            sample_point_num,
        )
    return True

def testSpeedForAnchorNum():
    gt_points_file_path = '/home/chli/chLi/Dataset/Famous/sample_pcd/bunny.npy'
    anchor_num_list = [10, 20, 50, 100, 200, 400]

    for i in range(len(anchor_num_list)):
        anchor_num = anchor_num_list[i]
        print('start testSpeed on anchor_num:', anchor_num)
        testSpeed(
            gt_points_file_path,
            400000,
            anchor_num,
            1000,
        )
    return True

if __name__ == '__main__':
    #testSpeedForSampleNum()
    testSpeedForAnchorNum()
