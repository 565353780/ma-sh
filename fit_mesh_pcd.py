import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import open3d as o3d

from ma_sh.Demo.pcd_trainer import demo as demo_train_pcd
from ma_sh.Data.mesh import Mesh
from ma_sh.Method.pcd import getPointCloud
from ma_sh.Method.path import createFileFolder


if __name__ == "__main__":
    home = os.environ["HOME"]
    shape_data_dict = {
        "BitAZ": {
            "mesh": home + "/chLi/Dataset/BitAZ/mesh/BitAZ.ply",
            "pcd": home + "/chLi/Dataset/BitAZ/pcd/BitAZ.ply",
        }
    }
    shape_id = "BitAZ"

    mesh_file_path = shape_data_dict[shape_id]["mesh"]
    pcd_file_path = shape_data_dict[shape_id]["pcd"]
    gt_points_num = 400000
    save_root_folder_path = home + "/chLi/Results/ma-sh/MeshTrainer/" + shape_id + "/"

    if not os.path.exists(pcd_file_path):
        mesh = Mesh(mesh_file_path)

        assert mesh.isValid()

        points = mesh.toSamplePoints(gt_points_num)
        assert isinstance(points, np.ndarray)

        pcd = getPointCloud(points)
        createFileFolder(pcd_file_path)
        o3d.io.write_point_cloud(pcd_file_path, pcd, write_ascii=True)

    demo_train_pcd(
        pcd_file_path,
        anchor_num=1600,
        mask_degree_max=3,
        sh_degree_max=2,
        save_freq=-1,
        save_log_folder_path=save_root_folder_path + "logs/" + shape_id + "/",
        save_result_folder_path=save_root_folder_path + "results/" + shape_id + "/",
    )
