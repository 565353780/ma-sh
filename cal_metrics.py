import sys
sys.path.append('../wn-nc/')

import torch

from ma_sh.Metric.compare import recordMetrics

if __name__ == '__main__':
    gt_data_folder_path = '/home/chli/chLi/Dataset/Thingi10K/mesh/'
    query_data_folder_path = '/home/chli/chLi/Dataset/Thingi10K/mesh_mash-1600anc/'
    gt_type = '.obj'
    query_type = '.npy'
    query_mode = 'mash'
    save_metric_file_path = './output/metrics/Thingi10K_ours.npy'
    sample_point_num = 50000
    chamfer_order = 2.0
    fscore_thresh = 0.01
    dtype = torch.float32
    device = 'cuda'

    recordMetrics(
        gt_data_folder_path=gt_data_folder_path,
        query_data_folder_path=query_data_folder_path,
        gt_type=gt_type,
        query_type=query_type,
        query_mode=query_mode,
        save_metric_file_path=save_metric_file_path,
        sample_point_num=sample_point_num,
        chamfer_order=chamfer_order,
        fscore_thresh=fscore_thresh,
        dtype=dtype,
        device=device,
    )
