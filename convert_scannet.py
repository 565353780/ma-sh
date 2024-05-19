import os
import torch
import numpy as np

from ma_sh.Data.mesh import Mesh
from ma_sh.Method.path import createFileFolder
from ma_sh.Module.trainer import Trainer


def demo():
    anchor_num = 2000
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

    gt_points_num = 400000

    scannet_dataset_folder_path = '/home/chli/chLi/Dataset/ScanNet/scans/'

    scene_name_list = os.listdir(scannet_dataset_folder_path)
    scene_name_list.sort()

    for scene_name in scene_name_list:
        mesh_id = scene_name + '_vh_clean_2'
        mesh_file_path = scannet_dataset_folder_path + scene_name + '/' + mesh_id + '.ply'
        if not os.path.exists(mesh_file_path):
            continue

        normalized_mesh_file_path = '/home/chli/chLi/Dataset/NormalizedMesh/ScanNet/' + scene_name + '/' + mesh_id + '.ply'
        if not os.path.exists(normalized_mesh_file_path):
            createFileFolder(normalized_mesh_file_path)
            mesh = Mesh(mesh_file_path)

            min_bound = np.min(mesh.vertices, axis=0)
            max_bound = np.max(mesh.vertices, axis=0)
            length = np.max(max_bound - min_bound)
            scale = 0.9 / length
            center = (min_bound + max_bound) / 2.0

            mesh.vertices = (mesh.vertices - center) * scale

            mesh.save(normalized_mesh_file_path, True)

        sampled_pcd_file_path = '/home/chli/chLi/Dataset/SampledPcd_Manifold/ScanNet/' + scene_name + '/' + mesh_id + '.npy'
        if not os.path.exists(sampled_pcd_file_path):
            createFileFolder(sampled_pcd_file_path)
            mesh = Mesh(normalized_mesh_file_path)
            points = mesh.toSamplePoints(gt_points_num)
            np.save(sampled_pcd_file_path, points)

        save_result_folder_path = "/home/chli/Nutstore Files/MASH-Materials/scene_materials/Anc-" + str(anchor_num) + '/' + scene_name + '/'
        if os.path.exists(save_result_folder_path):
            continue

        save_log_folder_path = "./logs/Anc-" + str(anchor_num) + '/' + scene_name + '/'

        save_params_file_path = "./output/Anc-" + str(anchor_num) + '/' + scene_name + ".npy"
        save_pcd_file_path = "./output/Anc-" + str(anchor_num) + '/' + scene_name + ".ply"
        overwrite = True
        print_progress = True

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
            save_result_folder_path,
            save_log_folder_path,
        )

        print('start fitting Famous:', scene_name)
        trainer.loadGTPointsFile(sampled_pcd_file_path)
        trainer.autoTrainMash(gt_points_num)
        trainer.mash.saveParamsFile(save_params_file_path, overwrite)
        trainer.mash.saveAsPcdFile(save_pcd_file_path, overwrite, print_progress)
    return True

if __name__ == "__main__":
    demo()
