import sys
sys.path.append('../wn-nc/')
sys.path.append('../siggraph-rebuttal/')

import torch
import numpy as np

from siggraph_rebuttal.Method.table import toTableStr

from ma_sh.Method.metric import recordMetrics, toMeanMetric

def toTableStrFromFile(metric_file_path: str) -> str:
    metric_dict = np.load(metric_file_path, allow_pickle=True).item()

    data = []

    title = ['Method', 'Ours']
    data.append(title)

    for value in metric_dict.values():
        for metric_key in value.keys():
            data.append([metric_key, toMeanMetric(metric_dict, metric_key)])
        break

    return toTableStr(data)

if __name__ == '__main__':
    gt_data_folder_path = '/home/chli/chLi/Dataset/Thingi10K/mesh/'
    query_data_folder_path = '/home/chli/chLi/Dataset/Thingi10K/mesh_mash-1600anc/'
    gt_type = '.obj'
    query_type = '.npy'
    query_mode = 'mash'
    save_metric_file_path = './output/metrics/Thingi10K_ours.npy'
    sample_point_num = 50000
    chamfer_order_list = [1.0, 2.0]
    fscore_thresh_list = [0.01, 0.001, 0.0001]
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
        chamfer_order_list=chamfer_order_list,
        fscore_thresh_list=fscore_thresh_list,
        dtype=dtype,
        device=device,
    )

    table_str = toTableStrFromFile(save_metric_file_path)

    print("Table:")
    print(table_str)
