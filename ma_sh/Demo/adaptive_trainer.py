import os
import torch
import numpy as np
import open3d as o3d

from ma_sh.Config.custom_path import mesh_file_path_dict
from ma_sh.Method.pcd import getPointCloud
from ma_sh.Method.render import renderGeometries
from ma_sh.Module.adaptive_trainer import AdaptiveTrainer
from ma_sh.Module.timer import Timer


def demo():
    init_anchor_num = 40
    mask_degree_max = 3
    sh_degree_max = 2
    mask_boundary_sample_num = 90
    sample_polar_num = 1000
    sample_point_scale = 0.8
    use_inv = True
    idx_dtype = torch.int64
    dtype = torch.float32
    device = "cuda:0"
    max_fit_error = 1e-4

    lr = 2e-3
    min_lr = 1e-3
    warmup_step_num = 80
    warmup_epoch = 4
    factor = 0.8
    patience = 2

    render = False
    render_freq = 1
    render_init_only = False
    save_freq = 1

    gt_points_num = 400000

    save_result_folder_path = None
    save_log_folder_path = None

    save_result_folder_path = 'auto'
    save_log_folder_path = 'auto'

    if False:
        mesh_name = "linux_airplane"
        mesh_file_path = mesh_file_path_dict[mesh_name]
    elif False:
        dataset_folder_path = "/home/chli/chLi2/Dataset/NormalizedMesh/ShapeNet/02691156/"
        mesh_filename_list = os.listdir(dataset_folder_path)
        mesh_filename_list.sort()
        mesh_filename = mesh_filename_list[0]
        mesh_file_path = dataset_folder_path + mesh_filename
        mesh_name = mesh_filename.split('.obj')[0]
    else:
        mesh_name = '03001627/1016f4debe988507589aae130c1f06fb'
        mesh_name = '02691156/1066b65c30d153e04c3a35cee92bb95b'

    save_params_file_path = "./output/" + mesh_name + "_Anc-" + str(init_anchor_num) + ".npy"
    save_pcd_file_path = "./output/" + mesh_name + "_Anc-" + str(init_anchor_num) + ".ply"
    overwrite = True
    print_progress = True

    adaptive_trainer = AdaptiveTrainer(
        init_anchor_num,
        mask_degree_max,
        sh_degree_max,
        mask_boundary_sample_num,
        sample_polar_num,
        sample_point_scale,
        use_inv,
        idx_dtype,
        dtype,
        device,
        max_fit_error,
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

    if False:
        adaptive_trainer.loadMeshFile(mesh_file_path)
    else:
        gt_points_file_path = "/home/chli/chLi2/Dataset/SampledPcd_Manifold/ShapeNet/04090263/22d2782aa73ea40960abd8a115f9899.npy"
        gt_points_file_path = "/home/chli/chLi2/Dataset/SampledPcd_Manifold/ShapeNet/03001627/46e1939ce6ee14d6a4689f3cf5c22e6.npy"
        gt_points_file_path = "/home/chli/chLi2/Dataset/SampledPcd_Manifold/ShapeNet/03001627/1b8e84935fdc3ec82be289de70e8db31.npy"
        gt_points_file_path = "/home/chli/chLi2/Dataset/SampledPcd_Manifold/ShapeNet/03001627/e71d05f223d527a5f91663a74ccd2338.npy"
        gt_points_file_path = "/home/chli/chLi2/Dataset/SampledPcd_Manifold/ShapeNet/" + mesh_name + ".npy"
        #gt_points_file_path = "../mvs-former/output/" + mesh_name + "/" + mesh_name + ".ply"
        # gt_points_file_path = "/Users/fufu/Downloads/model_normalized_obj.npy"
        adaptive_trainer.loadGTPointsFile(gt_points_file_path, gt_points_num)

    gt_points = np.load(gt_points_file_path)
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_points)
    o3d.io.write_point_cloud('./output/gt_points.ply', gt_pcd)

    timer = Timer()
    adaptive_trainer.autoTrainMash(gt_points_num)
    adaptive_trainer.saveMashFile(save_params_file_path, overwrite)
    adaptive_trainer.saveAsPcdFile(save_pcd_file_path, overwrite)

    if adaptive_trainer.o3d_viewer is not None:
        adaptive_trainer.o3d_viewer.run()

    # adaptive_trainer.mash.renderSamplePoints()

    # render final result
    if False:
        mesh_abb_length = 2.0 * adaptive_trainer.mesh.toABBLength()
        if mesh_abb_length == 0:
            mesh_abb_length = 1.1

        gt_pcd = getPointCloud(adaptive_trainer.gt_points)
        gt_pcd.translate([-mesh_abb_length, 0, 0])

        detect_points = torch.vstack(adaptive_trainer.mash.toSamplePoints()[:2]).detach().clone().cpu().numpy()
        print("detect_points:", detect_points.shape)
        print(
            "inner points for each anchor:",
            detect_points.shape[0] / anchor_num - mask_boundary_sample_num,
        )
        pcd = getPointCloud(detect_points)

        # adaptive_trainer.mesh.paintJetColorsByPoints(detect_points)
        mesh = adaptive_trainer.mesh.toO3DMesh()
        mesh.translate([mesh_abb_length, 0, 0])

        renderGeometries([gt_pcd, pcd, mesh])

    print('finish training, spend time :', timer.now())
    return True
