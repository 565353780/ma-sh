import os
import torch
import trimesh
import numpy as np
import open3d as o3d

import mash_cpp

from ma_sh.Method.data import toNumpy
from ma_sh.Method.path import createFileFolder, removeFile, renameFile

@torch.no_grad()
def compareCD():
    dataset_folder_path = '/home/chli/chLi/Dataset/'
    sample_point_num = 50000
    dtype = torch.float32
    device = 'cuda'
    save_metric_file_path = './output/metric.npy'

    metric_dict = {}

    if os.path.exists(save_metric_file_path):
        metric_dict = np.load(save_metric_file_path, allow_pickle=True).item()

    gt_folder_path = dataset_folder_path + 'NormalizedMesh/'
    gt_type = '.obj'

    mash_folder_path = dataset_folder_path + 'Mash_Recon/'
    mash_type = '.ply'

    pgr_folder_path = dataset_folder_path + 'PGR_Recon_2048/'
    pgr_type = '.ply'

    dataset_name_list = os.listdir(mash_folder_path)
    dataset_name_list.sort()

    common_shape_num = 0

    for dataset_name in dataset_name_list:
        if dataset_name not in metric_dict.keys():
            metric_dict[dataset_name] = {}

        gt_dataset_folder_path = gt_folder_path + dataset_name + '/'

        mash_dataset_folder_path = mash_folder_path + dataset_name + '/'

        pgr_dataset_folder_path = pgr_folder_path + dataset_name + '/'
        if not os.path.exists(pgr_dataset_folder_path):
            continue

        category_list = os.listdir(mash_dataset_folder_path)
        category_list.sort()

        for category in category_list:
            if category not in metric_dict[dataset_name].keys():
                metric_dict[dataset_name][category] = {}

            gt_category_folder_path = gt_dataset_folder_path + category + '/'

            mash_category_folder_path = mash_dataset_folder_path + category + '/'

            pgr_category_folder_path = pgr_dataset_folder_path + category + '/'
            if not os.path.exists(pgr_category_folder_path):
                continue

            mash_mesh_filename_list = os.listdir(mash_category_folder_path)
            mash_mesh_filename_list.sort()

            for mesh_filename in mash_mesh_filename_list:
                mesh_id = mesh_filename.split(mash_type)[0]

                if mesh_id not in metric_dict[dataset_name][category].keys():
                    metric_dict[dataset_name][category][mesh_id] = {}

                gt_mesh_file_path = gt_category_folder_path + mesh_id + gt_type

                mash_mesh_file_path = mash_category_folder_path + mesh_id + mash_type

                pgr_mesh_file_path = pgr_category_folder_path + mesh_id + pgr_type
                if not os.path.exists(pgr_mesh_file_path):
                    continue

                common_shape_num += 1
                is_update = False

                gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_file_path)
                gt_sample_pcd = gt_mesh.sample_points_uniformly(sample_point_num)
                gt_pts = torch.from_numpy(np.asarray(gt_sample_pcd.points)).unsqueeze(0).type(dtype).to(device)

                if 'mash_cd' not in metric_dict[dataset_name][category][mesh_id].keys():
                    mash_mesh = trimesh.load_mesh(mash_mesh_file_path)
                    mash_sample_pts = mash_mesh.sample(sample_point_num)
                    mash_pts = torch.from_numpy(np.asarray(mash_sample_pts)).type(dtype).to(device)
                    mash_dist1, mash_dist2 = mash_cpp.toChamferDistanceLoss(mash_pts, gt_pts)
                    mash_cd = toNumpy(mash_dist1 + mash_dist2)
                    metric_dict[dataset_name][category][mesh_id]['mash_cd'] = mash_cd
                    is_update = True

                if 'pgr_cd' not in metric_dict[dataset_name][category][mesh_id].keys():
                    pgr_mesh = trimesh.load_mesh(pgr_mesh_file_path)
                    pgr_sample_pts = pgr_mesh.sample(sample_point_num)
                    pgr_pts = torch.from_numpy(np.asarray(pgr_sample_pts)).type(dtype).to(device)
                    pgr_dist1, pgr_dist2 = mash_cpp.toChamferDistanceLoss(pgr_pts, gt_pts)
                    pgr_cd = toNumpy(pgr_dist1 + pgr_dist2)
                    metric_dict[dataset_name][category][mesh_id]['pgr_cd'] = pgr_cd
                    is_update = True

                print(common_shape_num, '--> shape_id:', mesh_id, '\t mash_cd:', metric_dict[dataset_name][category][mesh_id]['mash_cd'], '\t pgr_cd:', metric_dict[dataset_name][category][mesh_id]['pgr_cd'])

                if is_update:
                    tmp_save_metric_file_path = save_metric_file_path.replace('.npy', '_tmp.npy')
                    createFileFolder(tmp_save_metric_file_path)

                    np.save(tmp_save_metric_file_path, metric_dict, allow_pickle=True)
                    removeFile(save_metric_file_path)
                    renameFile(tmp_save_metric_file_path, save_metric_file_path)

    return True
