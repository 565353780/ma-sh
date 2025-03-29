import sys
sys.path.append('../wn-nc')

import os
import numpy as np
import open3d as o3d

from ma_sh.Method.pcd import getPointCloud
from ma_sh.Module.single_anchor_trainer import SingleAnchorTrainer

def demo():
    gt_pcd_file_path = '/home/chli/chLi/Dataset/Famous/sample_pcd/bunny.ply'
    surface_position = np.array([0.350297, -0.048401, -0.072172])
    anchor_rel_position = np.array([0.05, -0.05, 0]) * 0.1
    pretrained_mash_file_path = '/home/chli/github/ASDF/ma-sh/output/single_anchor_fitting/mash/648_final_anc-1_mash.npy'
    vis_pretrained_mash = True

    if not os.path.exists(pretrained_mash_file_path):
        pretrained_mash_file_path = None
        vis_pretrained_mash = False

    single_anchor_trainer = SingleAnchorTrainer(
        mask_degree_max=0,
        sh_degree_max=2,
        use_inv=True,
        device='cuda',
        patience=4,
        render=vis_pretrained_mash,
        render_init_only=vis_pretrained_mash,
        save_result_folder_path='auto',
        save_log_folder_path='auto',
    )

    single_anchor_trainer.loadGTPointsFileOnly(gt_pcd_file_path)

    gt_pcd = getPointCloud(single_anchor_trainer.gt_points)

    anchor_position = surface_position + anchor_rel_position

    single_anchor_trainer.setMashPose(anchor_position, surface_position, 1e-6)
    if pretrained_mash_file_path is not None:
        single_anchor_trainer.mash.loadParamsFile(pretrained_mash_file_path)
    if vis_pretrained_mash:
        single_anchor_trainer.setFullMask()

    single_anchor_trainer.autoTrainMash()

    mash_pcd = single_anchor_trainer.mash.toSamplePcd()

    axis_custom = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    axis_custom.translate(anchor_position)

    o3d.visualization.draw_geometries([gt_pcd, axis_custom, mash_pcd])

    return True
