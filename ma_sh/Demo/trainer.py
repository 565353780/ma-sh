import sys

sys.path.append("../mash-occ-decoder")
import torch

try:
    from mash_occ_decoder.Dataset.sdf import SDFDataset
except:
    pass

from ma_sh.Config.custom_path import mesh_file_path_dict
from ma_sh.Method.pcd import getPointCloud
from ma_sh.Method.render import renderGeometries
from ma_sh.Module.trainer import Trainer


def demo():
    anchor_num = 40
    mask_degree_max = 4
    sh_degree_max = 6
    mask_boundary_sample_num = 18
    sample_polar_num = 100000
    sample_point_scale = 0.4
    use_inv = True
    idx_dtype = torch.int64
    dtype = torch.float64
    device = "cuda:0"

    warm_epoch_step_num = 10
    warm_epoch_num = 2000
    finetune_step_num = 200000
    lr = 2e-2
    weight_decay = 1e-10
    factor = 0.99
    patience = 400
    min_lr = 1e-4

    render = False
    render_freq = 1
    render_init_only = False

    gt_points_num = 400000

    save_result_folder_path = "auto"
    save_log_folder_path = "auto"

    if False:
        mesh_name = "linux_airplane"
        mesh_file_path = mesh_file_path_dict[mesh_name]
    else:
        dataset_root_folder_path = "/home/chli/chLi/Dataset/"
        sdf_dataset = SDFDataset(dataset_root_folder_path, "test")

        object_id = 2
        _, sdf_file_path = sdf_dataset.paths_list[object_id]

        mesh_name = "chair" + str(object_id)
        mesh_file_path = sdf_file_path.replace(
            sdf_dataset.sdf_folder_path + "ShapeNet/sdf/",
            "/home/chli/chLi/Dataset/ShapeNet/Core/ShapeNetCore.v2/",
        ).replace("_obj.npy", ".obj")
        mesh_file_path = sdf_file_path.replace(
            sdf_dataset.sdf_folder_path + "ShapeNet/sdf/",
            "/home/chli/chLi/Dataset/SDF/ShapeNet/manifold/",
        ).replace("_obj.npy", "_obj.obj")

    save_params_file_path = "./output/" + mesh_name + ".npy"
    save_pcd_file_path = "./output/" + mesh_name + ".ply"
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
        warm_epoch_step_num,
        warm_epoch_num,
        finetune_step_num,
        lr,
        weight_decay,
        factor,
        patience,
        min_lr,
        render,
        render_freq,
        render_init_only,
        save_result_folder_path,
        save_log_folder_path,
    )

    if False:
        trainer.loadMeshFile(mesh_file_path)
    else:
        gt_points_file_path = "/home/chli/chLi/Dataset/Mash/ShapeNet/pcd/04090263/22d2782aa73ea40960abd8a115f9899/models/model_normalized_obj.npy"
        # gt_points_file_path = "/Users/fufu/Downloads/model_normalized_obj.npy"
        trainer.loadGTPointsFile(gt_points_file_path)
    trainer.autoTrainMash(gt_points_num)
    trainer.mash.saveParamsFile(save_params_file_path, overwrite)
    trainer.mash.saveAsPcdFile(save_pcd_file_path, overwrite, print_progress)

    if trainer.o3d_viewer is not None:
        trainer.o3d_viewer.run()
    # trainer.mash.renderSamplePoints()

    mesh_abb_length = 2.0 * trainer.mesh.toABBLength()
    if mesh_abb_length == 0:
        mesh_abb_length = 1.1

    gt_pcd = getPointCloud(trainer.gt_points)
    gt_pcd.translate([-mesh_abb_length, 0, 0])

    detect_points = trainer.mash.toSamplePoints().detach().clone().cpu().numpy()
    print("detect_points:", detect_points.shape)
    print(
        "inner points for each anchor:",
        detect_points.shape[0] / anchor_num - mask_boundary_sample_num,
    )
    pcd = getPointCloud(detect_points)

    # trainer.mesh.paintJetColorsByPoints(detect_points)
    mesh = trainer.mesh.toO3DMesh()
    mesh.translate([mesh_abb_length, 0, 0])

    renderGeometries([gt_pcd, pcd, mesh])
    return True
