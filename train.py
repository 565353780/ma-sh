import os
import numpy as np
import open3d as o3d
from shutil import rmtree
# from torch import profiler

from ma_sh.Demo.trainer import demo as demo_train
from ma_sh.Method.pcd import getPointCloud

if __name__ == "__main__":
    '''
    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA],
        on_trace_ready=profiler.tensorboard_trace_handler('./logs/')
    ) as prof:
        demo_train(400)

    print(prof.key_averages().table(sort_by="cpu_time_total"))
    exit()
    '''

    shapenet_shape_id_list = [
        '02691156/1066b65c30d153e04c3a35cee92bb95b',
        '03001627/e71d05f223d527a5f91663a74ccd2338',
    ]
    objaverse_shape_id_list = [
        # '000-091/bf193e241b2f48f0bd2208f89e38fae8',
        # '000-091/91979ad79916460d92c7697464f2b5f4',
        # '000-091/d4efa3e396274421b07b2fa4314c60bb',
        # '000-091/97c493d5c7a443b89229e5f7edb3ae4a',
        '000-091/01fcb4e4c36548ca86624b63dfc6b255',
        '000-091/9df219962230449caa4c95a60feb0c9e',
    ]
    anchor_num_list = [10, 20, 50, 100, 200, 400]
    save_freq = 1
    render_only = False

    shapenet_gt_points_file_path_dict = {
        shape_id: '/home/chli/chLi2/Dataset/SampledPcd_Manifold/ShapeNet/' + shape_id + '.npy' for shape_id in shapenet_shape_id_list
    }
    objaverse_gt_points_file_path_dict = {
        shape_id: '/home/chli/chLi/Dataset/Objaverse_82K/manifold_pcd/' + shape_id + '.npy' for shape_id in objaverse_shape_id_list
    }

    gt_points_file_path_dict = {}
    gt_points_file_path_dict.update(shapenet_gt_points_file_path_dict)
    gt_points_file_path_dict.update(objaverse_gt_points_file_path_dict)

    for shape_id, gt_points_file_path in gt_points_file_path_dict.items():
        if not os.path.exists(gt_points_file_path):
            continue

        shape_id = shape_id.replace('/', '_')

        if render_only:
            points = np.load(gt_points_file_path)
            pcd = getPointCloud(points)
            print('shape_id:', shape_id)
            o3d.visualization.draw_geometries([pcd])
            continue

        for anchor_num in anchor_num_list:
            save_log_folder_path = '/home/chli/chLi/Results/ma-sh/logs/fixed/' + shape_id + '/anchor-' + str(anchor_num) + '/'
            save_result_folder_path = '/home/chli/chLi/Results/ma-sh/output/fit/fixed/' + shape_id + '/anchor-' + str(anchor_num) + '/'

            if os.path.exists(save_result_folder_path + 'mash/'):
                continue

            if os.path.exists(save_log_folder_path):
                rmtree(save_log_folder_path)
            if os.path.exists(save_result_folder_path):
                rmtree(save_result_folder_path)

            demo_train(gt_points_file_path,
                       anchor_num,
                       save_freq,
                       save_log_folder_path,
                       save_result_folder_path)
