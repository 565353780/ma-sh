import os
import torch
import trimesh
import numpy as np
import open3d as o3d

import mash_cpp

@torch.no_grad()
def compareCD():
    dataset_folder_path = '/home/chli/chLi/Dataset/'
    sample_point_num = 10000
    dtype = torch.float32
    device = 'cuda'

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
        gt_dataset_folder_path = gt_folder_path + dataset_name + '/'

        mash_dataset_folder_path = mash_folder_path + dataset_name + '/'

        pgr_dataset_folder_path = pgr_folder_path + dataset_name + '/'
        if not os.path.exists(pgr_dataset_folder_path):
            continue

        category_list = os.listdir(mash_dataset_folder_path)
        category_list.sort()

        for category in category_list:
            gt_category_folder_path = gt_dataset_folder_path + category + '/'

            mash_category_folder_path = mash_dataset_folder_path + category + '/'

            pgr_category_folder_path = pgr_dataset_folder_path + category + '/'
            if not os.path.exists(pgr_category_folder_path):
                continue

            mash_mesh_filename_list = os.listdir(mash_category_folder_path)
            mash_mesh_filename_list.sort()

            pgr_mesh_filename_list = os.listdir(pgr_category_folder_path)

            for mesh_filename in mash_mesh_filename_list:
                mesh_id = mesh_filename.split(mash_type)[0]

                gt_mesh_file_path = gt_category_folder_path + mesh_id + gt_type

                mash_mesh_file_path = mash_category_folder_path + mesh_id + mash_type

                pgr_mesh_file_path = pgr_category_folder_path + mesh_id + pgr_type
                if not os.path.exists(pgr_mesh_file_path):
                    continue

                common_shape_num += 1

                gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_file_path)
                mash_mesh = o3d.io.read_triangle_mesh(mash_mesh_file_path)
                pgr_mesh = o3d.io.read_triangle_mesh(pgr_mesh_file_path)

                gt_sample_pcd = gt_mesh.sample_points_uniformly(sample_point_num)
                mash_sample_pcd = mash_mesh.sample_points_uniformly(sample_point_num)
                pgr_sample_pcd = pgr_mesh.sample_points_uniformly(sample_point_num)

                gt_pts = torch.from_numpy(np.asarray(gt_sample_pcd.points)).unsqueeze(0).type(dtype).to(device)
                mash_pts = torch.from_numpy(np.asarray(mash_sample_pcd.points)).type(dtype).to(device)
                pgr_pts = torch.from_numpy(np.asarray(pgr_sample_pcd.points)).type(dtype).to(device)

                mash_dist1, mash_dist2 = mash_cpp.toChamferDistanceLoss(mash_pts, gt_pts)
                mash_cd = mash_dist1 + mash_dist2

                pgr_dist1, pgr_dist2 = mash_cpp.toChamferDistanceLoss(pgr_pts, gt_pts)
                pgr_cd = pgr_dist1 + pgr_dist2

                print('shape_id:', mesh_id, '    mash_cd:', mash_cd, 'pgr_cd:', pgr_cd)

                print('solved common shape num:', common_shape_num)
