import os
import torch
import functools
import numpy as np
import open3d as o3d
from copy import deepcopy

from ma_sh.Data.mesh import Mesh
from ma_sh.Method.data import toNumpy
from ma_sh.Method.pcd import getPointCloud
from ma_sh.Model.mash import Mash
from ma_sh.Module.o3d_viewer import O3DViewer


def compare(str_a: str, str_b: str) -> int:
    id_a = int(str_a.split("_")[0])
    id_b = int(str_b.split("_")[0])

    if id_a > id_b:
        return 1
    elif id_a == id_b:
        return 0
    return -1


def demo():
    # view mash dataset
    if False:
        mash_params_folder_path = "./output/dataset/"

        class_name_list = os.listdir(mash_params_folder_path)

        for class_name in class_name_list:
            mash_params_file_path = (
                mash_params_folder_path
                + class_name
                + "/models/model_normalized_obj.npy"
            )

            if not os.path.exists(mash_params_file_path):
                continue

            mash = Mash.fromParamsFile(mash_params_file_path, device="cpu")

            mash.renderSamplePoints()

    # view single mash
    if True:
        mash_folder_path = "/home/chli/chLi/Dataset/Objaverse_82K/mash/000-001/"
        if os.path.exists(mash_folder_path):
            mash_filename_list = os.listdir(mash_folder_path)

            for mash_filename in mash_filename_list:
                if ".npy" not in mash_filename:
                    continue

                mash_file_path = mash_folder_path + mash_filename

                if True:
                    mesh_file_path = "/home/chli/chLi/Dataset/Objaverse_82K/manifold/000-001/" + mash_filename[:-4] + '.obj'
                    if not os.path.exists(mesh_file_path):
                        continue

                    mesh = o3d.io.read_triangle_mesh(mesh_file_path)

                    mesh.compute_vertex_normals()
                    mesh.compute_triangle_normals()

                mash = Mash.fromParamsFile(mash_file_path, 90, 1000, 0.8, device="cuda")

                mash_pcd = mash.toSamplePcd()

                print("start show mash:", mash_filename)
                o3d.visualization.draw_geometries([mesh, mash_pcd])

    if False:
        mash_params_folder_path = "/Volumes/chLi/Dataset/Mash/ShapeNet/mash/"

        class_name_list = os.listdir(mash_params_folder_path)

        for class_name in class_name_list:
            class_folder_path = mash_params_folder_path + class_name + "/"

            model_id_list = os.listdir(class_folder_path)

            for i, model_id in enumerate(model_id_list):
                mash_file_path = (
                    class_folder_path + model_id + "/models/model_normalized_obj.npy"
                )

                mash = Mash.fromParamsFile(mash_file_path, 10, 10000, 0.4, device="cpu")

                mash.renderSamplePoints()

                if i >= 3:
                    break

    # view single training process
    if False:
        boundary_sample_num = 36
        inner_sample_num = 100
        fps_scale = 0.8
        view_freq = 4
        show_recon_only = False

        o3d_viewer = O3DViewer()
        o3d_viewer.createWindow()

        mash_root_folder_path = "./output/"

        mash_folename_list = os.listdir(mash_root_folder_path)
        mash_folename_list.sort()

        valid_mash_folder_path_list = []

        for mash_folename in mash_folename_list:
            mash_folder_path = mash_root_folder_path + mash_folename + "/"

            if not os.path.isdir(mash_folder_path) or not os.path.exists(
                mash_folder_path
            ):
                continue

            valid_mash_folder_path_list.append(mash_folder_path)

        mash_folder_path = valid_mash_folder_path_list[-1]
        print("start view:", mash_folder_path)

        mash_filename_list = os.listdir(mash_folder_path)
        mash_filename_list.sort(key=functools.cmp_to_key(compare))

        if show_recon_only:
            mash_file_path = mash_folder_path + mash_filename_list[-1]

            mash = Mash.fromParamsFile(
                mash_file_path,
                boundary_sample_num,
                inner_sample_num,
                fps_scale,
            )
            boundary_pts, inner_pts, inner_idxs = mash.toSamplePoints()
            points = toNumpy(torch.vstack([boundary_pts, inner_pts]))

            mash.renderSamplePatches()
            return True

        for i, mash_filename in enumerate(mash_filename_list):
            if i + 1 != len(mash_filename_list):
                if (i + 1) % view_freq != 0:
                    continue

            mash_file_path = mash_folder_path + mash_filename

            mash = Mash.fromParamsFile(
                mash_file_path,
                boundary_sample_num,
                inner_sample_num,
                fps_scale,
            )
            boundary_pts, inner_pts, inner_idxs = mash.toSamplePoints()
            points = toNumpy(torch.vstack([boundary_pts, inner_pts]))

            pcd = getPointCloud(points)

            o3d_viewer.clearGeometries()
            o3d_viewer.addGeometry(pcd)

            print("now render is", i + 1, "/", len(mash_filename_list))

            o3d_viewer.update()

        o3d_viewer.run()
    return True
