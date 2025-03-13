import sys

sys.path.append('../wn-nc/')
sys.path.append('../siggraph-rebuttal/')

import os
import torch
import numpy as np

from siggraph_rebuttal.Method.table import toTableStr

from ma_sh.Config.custom_path import toDatasetRootPath
from ma_sh.Data.metric import (
    Coverage_Data,
    SAMPLE_Data,
    ShapeNet_NonUniform_1024_Data,
    ShapeNet_NonUniform_2048_Data,
    Timing_Data,
    ShapeNet_Data,
    ShapeNet_NonUniform_Data,
    Thingi10K_Data,
)
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

def demoEvalData():
    dataset_root_path = toDatasetRootPath()
    assert dataset_root_path is not None

    gt_data_folder_path = dataset_root_path + 'Thingi10K/mesh/'
    query_data_folder_path = dataset_root_path + 'Thingi10K/mesh_mash-1600anc/'
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
    return True

def createRecordForSetting(setting: str) -> bool:
    dataset_root_path = toDatasetRootPath()
    assert dataset_root_path is not None

    save_metric_file_path = './output/metrics/' + setting + '.npy'

    if setting == 'ShapeNet_MASH':
        recordMetrics(
            dataset_root_path + 'ShapeNet/manifold/',
            dataset_root_path + 'ShapeNet/manifold_mash/',
            '.obj', '.npy', 'mash', save_metric_file_path,
        )

    if setting == 'ShapeNet_MASHMesh':
        recordMetrics(
            dataset_root_path + 'ShapeNet/manifold/',
            dataset_root_path + 'ShapeNet/manifold_mash/',
            '.obj', '.npy', 'mashmesh', save_metric_file_path,
        )

    if setting == 'ShapeNet_NonUniform_MASH':
        recordMetrics(
            dataset_root_path + 'ShapeNet/manifold/',
            dataset_root_path + 'ShapeNet/manifold_mash-4096_random-10_noise-0-002-400anc/',
            '.obj', '.npy', 'mash', save_metric_file_path,
        )

    if setting == 'ShapeNet_NonUniform_MASHMesh':
        recordMetrics(
            dataset_root_path + 'ShapeNet/manifold/',
            dataset_root_path + 'ShapeNet/manifold_mash-4096_random-10_noise-0-002-400anc/',
            '.obj', '.npy', 'mashmesh', save_metric_file_path,
        )

    if setting == 'ShapeNet_NonUniform_2048_MASH':
        recordMetrics(
            dataset_root_path + 'ShapeNet/manifold_test/',
            dataset_root_path + 'ShapeNet/manifold_mash-2048_random-10-400anc/03001627/',
            '.obj', '.npy', 'mash', save_metric_file_path,
        )

    if setting == 'ShapeNet_NonUniform_2048_MASH-100anc':
        recordMetrics(
            dataset_root_path + 'ShapeNet/manifold_test/',
            dataset_root_path + 'ShapeNet/manifold_mash-2048_random-10-100anc/03001627/',
            '.obj', '.npy', 'mash', save_metric_file_path,
        )

    if setting == 'ShapeNet_NonUniform_1024_MASH':
        recordMetrics(
            dataset_root_path + 'ShapeNet/manifold_test/',
            dataset_root_path + 'ShapeNet/manifold_mash-1024_random-10-400anc/03001627/',
            '.obj', '.npy', 'mash', save_metric_file_path,
        )

    if setting == 'ShapeNet_NonUniform_1024_MASH-50anc':
        recordMetrics(
            dataset_root_path + 'ShapeNet/manifold_test/',
            dataset_root_path + 'ShapeNet/manifold_mash-1024_random-10-50anc/03001627/',
            '.obj', '.npy', 'mash', save_metric_file_path,
        )

    if setting == 'ShapeNet_NonUniform_ARONet':
        recordMetrics(
            dataset_root_path + 'ShapeNet/manifold/',
            '/home/chli/github/ASDF/aro-net/output/ShapeNet_NonUniform/',
            '.obj', '.obj', 'mesh', save_metric_file_path,
        )

    if setting == 'ShapeNet_NonUniform_2048_ARONet':
        recordMetrics(
            dataset_root_path + 'ShapeNet/manifold_test/',
            '/home/chli/github/ASDF/aro-net/output/ShapeNet_2048/03001627/',
            '.obj', '.obj', 'mesh', save_metric_file_path,
        )

    if setting == 'ShapeNet_NonUniform_1024_ARONet':
        recordMetrics(
            dataset_root_path + 'ShapeNet/manifold_test/',
            '/home/chli/github/ASDF/aro-net/output/ShapeNet_1024/03001627/',
            '.obj', '.obj', 'mesh', save_metric_file_path,
        )

    if setting == 'ShapeNet_NonUniform_PGR':
        recordMetrics(
            dataset_root_path + 'ShapeNet/manifold_test/',
            '/home/chli/github/ASDF/param-gauss-recon/output/recon/k_7_min_0.0015_max_0.015_alpha_1.05_depth_min_1_depth_max_1/',
            '.obj', '.ply', 'mesh', save_metric_file_path,
        )

    if setting == 'ShapeNet_NonUniform_2048_PGR':
        recordMetrics(
            dataset_root_path + 'ShapeNet/manifold_test/',
            '/home/chli/github/ASDF/param-gauss-recon/output/recon/2048/',
            '.obj', '.ply', 'mesh', save_metric_file_path,
        )

    if setting == 'ShapeNet_NonUniform_1024_PGR':
        recordMetrics(
            dataset_root_path + 'ShapeNet/manifold_test/',
            '/home/chli/github/ASDF/param-gauss-recon/output/recon/1024/',
            '.obj', '.ply', 'mesh', save_metric_file_path,
        )

    if setting == 'Thingi10K_MASH_400':
        recordMetrics(
            dataset_root_path + 'Thingi10K/mesh/',
            dataset_root_path + 'Thingi10K/mesh_mash-400anc/',
            '.obj', '.npy', 'mash', save_metric_file_path,
        )

    if setting == 'Thingi10K_MASHMesh_400':
        recordMetrics(
            dataset_root_path + 'Thingi10K/mesh/',
            dataset_root_path + 'Thingi10K/mesh_mash-400anc/',
            '.obj', '.npy', 'mashmesh', save_metric_file_path,
        )

    if setting == 'Thingi10K_MASH_1600':
        recordMetrics(
            dataset_root_path + 'Thingi10K/mesh/',
            dataset_root_path + 'Thingi10K/mesh_mash-1600anc/',
            '.obj', '.npy', 'mash', save_metric_file_path,
        )

    if setting == 'Thingi10K_ARONet':
        recordMetrics(
            dataset_root_path + 'Thingi10K/mesh/',
            '/home/chli/github/ASDF/aro-net/output/Thingi10K_400k/',
            '.obj', '.obj', 'mesh', save_metric_file_path,
        )

    if setting == 'Thingi10K_PGR':
        recordMetrics(
            dataset_root_path + 'Thingi10K/mesh/',
            '/home/chli/github/ASDF/param-gauss-recon/output/recon/k_7_min_0.0015_max_0.015_alpha_1.05_depth_min_1_depth_max_1/',
            '.obj', '.ply', 'mesh', save_metric_file_path,
        )

    if setting == 'SAMPLE_MASH':
        recordMetrics(
            dataset_root_path + 'Objaverse_82K/manifold/',
            '/home/chli/chLi/Results/ma-sh/output/fit/mash_anc400_sh2/',
            '.obj', '.npy', 'mash', save_metric_file_path,
        )

    if setting == 'SAMPLE_SIMPLEMASH':
        recordMetrics(
            dataset_root_path + 'Objaverse_82K/manifold/',
            '/home/chli/chLi/Results/ma-sh/output/fit/simple_mash_anc400_sh2/',
            '.obj', '.npy', 'mash', save_metric_file_path,
        )
    return True

if __name__ == '__main__':
    dataset_root_path = toDatasetRootPath()
    assert dataset_root_path is not None

    setting = '1ShapeNet_NonUniform_2048_MASH-100anc'

    createRecordForSetting(setting)

    save_metric_file_path = './output/metrics/' + setting + '.npy'

    if os.path.exists(save_metric_file_path):
        table_str = toTableStrFromFile(save_metric_file_path)
        print("Table:")
        print(table_str)
    else:
        print('==== Timing_Data ====')
        print(toTableStr(Timing_Data))
        print('==== ShapeNet_Data ====')
        print(toTableStr(ShapeNet_Data))
        print('==== ShapeNet_NonUniform_Data ====')
        print(toTableStr(ShapeNet_NonUniform_Data))
        print('==== ShapeNet_NonUniform_2048_Data ====')
        print(toTableStr(ShapeNet_NonUniform_2048_Data))
        print('==== ShapeNet_NonUniform_1024_Data ====')
        print(toTableStr(ShapeNet_NonUniform_1024_Data))
        print('==== Thingi10K_Data ====')
        print(toTableStr(Thingi10K_Data))
        print('==== SAMPLE_Data ====')
        print(toTableStr(SAMPLE_Data))
        print('==== Coverage_Data ====')
        print(toTableStr(Coverage_Data))
