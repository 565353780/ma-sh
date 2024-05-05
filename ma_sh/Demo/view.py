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
        mash_params_file_path = "/Users/fufu/Downloads/Mash/chairs/4.npy"

        mash_folder_path = "../Results/output-mash/"
        mash_filename_list = os.listdir(mash_folder_path)

        for mash_filename in mash_filename_list:
            if ".npy" not in mash_filename:
                continue

            print("start show mash:", mash_filename)
            mash_file_path = mash_folder_path + mash_filename

            if False:
                mesh_file_path = "./output/25d40c79ac57891cfebad4f49b26ec52.obj"
                mesh = o3d.io.read_triangle_mesh(mesh_file_path)

                mesh.compute_vertex_normals()
                mesh.compute_triangle_normals()

                mesh_pcd = mesh.sample_points_poisson_disk(100000)
                o3d.io.write_point_cloud(
                    "./output/test2_mesh_sample.ply", mesh_pcd, write_ascii=True
                )
                exit()

            try:
                mash = Mash.fromParamsFile(mash_file_path, 90, 1000, 0.8, device="cpu")
            except:
                pass

            mash.renderSamplePoints()
            exit()

            pcd = getPointCloud(toNumpy(torch.vstack(mash.toSamplePoints()[:2])))

            if True:
                pcd.estimate_normals()
                pcd.orient_normals_consistent_tangent_plane(10)
                o3d.visualization.draw_geometries([pcd], point_show_normal=True)
                exit()

            os.makedirs("./output/", exist_ok=True)
            o3d.io.write_point_cloud(
                "./output/" + mash_filename.replace(".npy", ".ply"),
                pcd,
                write_ascii=True,
            )
        exit()

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
    if True:
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
