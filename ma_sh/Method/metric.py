import os
import torch
import numpy as np
from typing import Tuple

from ma_sh.Data.mesh import Mesh
from ma_sh.Model.mash import Mash
import mash_cpp

from ma_sh.Method.path import createFileFolder, removeFile, renameFile
from ma_sh.Metric.chamfer import toChamfer
from ma_sh.Metric.fscore import toFScore
from ma_sh.Metric.nic import toNormalInConsistency


@torch.no_grad()
def getMetrics(
    gt_pts: np.ndarray,
    query_pts: np.ndarray,
    gt_normals: np.ndarray,
    query_normals: np.ndarray,
    chamfer_order_list: list = [1.0, 2.0],
    fscore_thresh_list: list = [0.01, 0.001],
    dtype=torch.float32,
    device: str = 'cuda',
) -> Tuple[dict, dict, torch.Tensor]:
    gt_pts = torch.from_numpy(np.asarray(gt_pts)).unsqueeze(0).type(dtype).to(device)
    query_pts = torch.from_numpy(np.asarray(query_pts)).unsqueeze(0).type(dtype).to(device)
    gt_normals = torch.from_numpy(np.asarray(gt_normals)).unsqueeze(0).type(dtype).to(device)
    query_normals = torch.from_numpy(np.asarray(query_normals)).unsqueeze(0).type(dtype).to(device)

    dists2_1, dists2_2, idxs_1 = mash_cpp.toChamferDistance(query_pts, gt_pts)[:3]

    pair_idxs = torch.stack([torch.arange(idxs_1.shape[1], dtype=torch.int64, device=device).expand(idxs_1.shape[0], idxs_1.shape[1]), idxs_1.type(torch.int64)], dim=-1)

    chamfer_dict = {}
    for chamfer_order in chamfer_order_list:
        chamfer_dict[str(chamfer_order)] = toChamfer(dists2_1, dists2_2, chamfer_order)[0].item()
    fscore_dict = {}
    for fscore_thresh in fscore_thresh_list:
        fscore_dict[str(fscore_thresh)] = toFScore(dists2_1, dists2_2, fscore_thresh)[0].item()
    nic = toNormalInConsistency(query_normals, gt_normals, pair_idxs)[0].item()

    return chamfer_dict, fscore_dict, nic

@torch.no_grad()
def recordMetrics(
    gt_data_folder_path: str,
    query_data_folder_path: str,
    gt_type: str,
    query_type: str,
    query_mode: str,
    save_metric_file_path: str,
    sample_point_num: int = 50000,
    chamfer_order_list: list = [1.0, 2.0],
    fscore_thresh_list: list = [0.01, 0.001],
    dtype=torch.float32,
    device: str = 'cuda',
) -> bool:
    assert query_mode in ['mesh', 'mash']

    metric_dict = {}

    if os.path.exists(save_metric_file_path):
        metric_dict = np.load(save_metric_file_path, allow_pickle=True).item()

    solved_shape_num = 0
    for root, _, files in os.walk(query_data_folder_path):
        for file in files:
            if not file.endswith(query_type):
                continue

            rel_file_basepath = os.path.relpath(root, query_data_folder_path) + '/' + file[:-4]

            if rel_file_basepath in metric_dict.keys():
                continue

            gt_file_path = gt_data_folder_path + rel_file_basepath + gt_type
            if not os.path.exists(gt_file_path):
                continue

            query_file_path = query_data_folder_path + rel_file_basepath + query_type
            if not os.path.exists(query_file_path):
                continue

            gt_mesh = Mesh(gt_file_path, sample_point_num)
            gt_pts = gt_mesh.sample_pts
            gt_normals = gt_mesh.sample_normals

            if query_mode == 'mesh':
                query_mesh = Mesh(query_file_path, sample_point_num)
                query_pts = query_mesh.sample_pts
                query_normals = query_mesh.sample_normals
            elif query_mode == 'mash':
                query_pcd = Mash.fromParamsFile(query_file_path, device='cuda').toSamplePcdWithWNNCNormals()
                query_pts = np.asarray(query_pcd.points)
                query_normals = np.asarray(query_pcd.normals)

            chamfer_dict, fscore_dict, nic = getMetrics(
                gt_pts=gt_pts,
                query_pts=query_pts,
                gt_normals=gt_normals,
                query_normals=query_normals,
                chamfer_order_list=chamfer_order_list,
                fscore_thresh_list=fscore_thresh_list,
                dtype=dtype,
                device=device,
            )

            metric_dict[rel_file_basepath] = {}
            for key, value in chamfer_dict.items():
                metric_dict[rel_file_basepath]['chamfer-' + key] = value
            for key, value in fscore_dict.items():
                metric_dict[rel_file_basepath]['fscore-' + key] = value
            metric_dict[rel_file_basepath]['nic'] = nic

            print('==== updated:', rel_file_basepath, '====')
            for key, item in metric_dict[rel_file_basepath].items():
                print(key, ':', item, '\t', end='')
            print()
            tmp_save_metric_file_path = save_metric_file_path.replace('.npy', '_tmp.npy')
            createFileFolder(tmp_save_metric_file_path)

            np.save(tmp_save_metric_file_path, metric_dict, allow_pickle=True)
            removeFile(save_metric_file_path)
            renameFile(tmp_save_metric_file_path, save_metric_file_path)

            solved_shape_num += 1
            print('solved_shape_num:', solved_shape_num)

    return True

def toMeanMetric(metric_dict: dict, metric_name: str) -> float:
    metric_list = []
    for value in metric_dict.values():
        if metric_name not in value.keys():
            continue
        metric_list.append(value[metric_name])

    if len(metric_list) == 0:
        return np.nan

    return np.mean(metric_list)
