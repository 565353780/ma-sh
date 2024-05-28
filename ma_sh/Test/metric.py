import os
import torch
import numpy as np
import open3d as o3d

import mash_cpp

from ma_sh.Config.shapenet import SHAPENET_NAME_DICT


@torch.no_grad()
def test():
    HOME = os.environ["HOME"]

    category_id_list = list(SHAPENET_NAME_DICT.keys())

    for category_id in category_id_list:
        category_id = category_id_list[6]
        mesh_folder_path = (
            HOME + "/chLi/Dataset/ManifoldMesh/ShapeNet/" + category_id + "/"
        )

        mesh_filename_list = os.listdir(mesh_folder_path)
        mesh_filename_list.sort()

        cd_list = []

        for mesh_filename in mesh_filename_list:
            mesh_file_path = mesh_folder_path + mesh_filename

            mesh = o3d.io.read_triangle_mesh(mesh_file_path)

            pts1 = (
                torch.from_numpy(
                    np.asarray(mesh.sample_points_poisson_disk(50000).points)
                )
                .float()
                .cuda()
            )
            pts2 = (
                torch.from_numpy(
                    np.asarray(mesh.sample_points_poisson_disk(50000).points)
                )
                .float()
                .unsqueeze(0)
                .cuda()
            )

            d1, d2 = mash_cpp.toChamferDistanceLoss(pts1, pts2)
            cd = d1 + d2

            cd_list.append(cd.item())

            print(SHAPENET_NAME_DICT[category_id], np.mean(cd_list).item())
    return True
