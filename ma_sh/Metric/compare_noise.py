import os
import torch
import numpy as np
import open3d as o3d
from shutil import copyfile

from ma_sh.Method.pcd import getPointCloud


@torch.no_grad()
def collectResults(gauss_sigma: float = 0.01):
    dataset_folder_path = '/home/chli/chLi/Dataset/'
    compare_resolution = '4000'
    noise_label = 'Noise_' + str(gauss_sigma).replace('.', '-')
    save_result_folder_path = './output/metric_manifold_result_' + noise_label + '/'

    print('start collect results for:', noise_label)

    os.makedirs(save_result_folder_path, exist_ok=True)

    gt_mesh_folder_path = dataset_folder_path + 'ManifoldMesh/'
    gt_mesh_type = '.obj'

    gt_pcd_folder_path = dataset_folder_path + 'SampledPcd_Manifold_' + noise_label + '/'
    gt_pcd_type = '.npy'

    mash_mesh_folder_path = dataset_folder_path + 'MashV4_Recon_' + noise_label + '/'
    mash_mesh_type = '.ply'

    pgr_mesh_folder_path = dataset_folder_path + 'PGR_Manifold_' + noise_label + '_Recon_' + compare_resolution + '/'
    pgr_mesh_type = '.ply'

    aro_mesh_folder_path = dataset_folder_path + 'ARONet_Manifold_' + noise_label + '_Recon_' + '2048' + '/'
    aro_mesh_type = '.obj'

    conv_mesh_folder_path = dataset_folder_path + 'ConvONet_Manifold_' + noise_label + '_Recon_' + '2048' + '/'
    conv_mesh_type = '.obj'

    dataset_name_list = os.listdir(mash_mesh_folder_path)
    dataset_name_list.sort()

    saved_num = 0
    for dataset_name in dataset_name_list:
        rel_file_path = dataset_name + '/'

        category_list = os.listdir(mash_mesh_folder_path + rel_file_path)

        for category in category_list:
            rel_file_path = dataset_name + '/' + category + '/'

            mesh_filename_list = os.listdir(mash_mesh_folder_path + rel_file_path)

            for mesh_filename in mesh_filename_list:
                mesh_id = mesh_filename.split('.ply')[0]

                rel_file_path = dataset_name + '/' + category + '/' + mesh_id

                current_save_result_folder_path = save_result_folder_path + rel_file_path + '/'

                os.makedirs(current_save_result_folder_path, exist_ok=True)

                gt_mesh_file_path = gt_mesh_folder_path + rel_file_path + gt_mesh_type
                gt_pcd_file_path = gt_pcd_folder_path + rel_file_path + gt_pcd_type
                mash_mesh_file_path = mash_mesh_folder_path + rel_file_path + mash_mesh_type
                pgr_mesh_file_path = pgr_mesh_folder_path + rel_file_path + pgr_mesh_type
                aro_mesh_file_path = aro_mesh_folder_path + rel_file_path + aro_mesh_type
                conv_mesh_file_path = conv_mesh_folder_path + rel_file_path + conv_mesh_type

                copyfile(gt_mesh_file_path, current_save_result_folder_path + 'gt_mesh' + gt_mesh_type)

                gt_points = np.load(gt_pcd_file_path)
                gt_pcd = getPointCloud(gt_points)
                sample_gt_pcd = gt_pcd.farthest_point_down_sample(int(compare_resolution))
                o3d.io.write_point_cloud(current_save_result_folder_path + 'gt_pcd.ply', sample_gt_pcd, write_ascii=True)

                copyfile(mash_mesh_file_path, current_save_result_folder_path + 'mash_mesh' + mash_mesh_type)
                copyfile(pgr_mesh_file_path, current_save_result_folder_path + 'pgr_mesh' + pgr_mesh_type)
                if os.path.exists(aro_mesh_file_path):
                    copyfile(aro_mesh_file_path, current_save_result_folder_path + 'aro_mesh' + aro_mesh_type)
                if os.path.exists(conv_mesh_file_path):
                    copyfile(conv_mesh_file_path, current_save_result_folder_path + 'conv_mesh' + conv_mesh_type)

                saved_num += 1
                print('solved shape num:', saved_num)

    print('collect results for:', noise_label, 'finished!')
    return True
