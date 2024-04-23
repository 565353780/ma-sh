import os
import numpy as np

from ma_sh.Data.mesh import Mesh
from ma_sh.Method.pcd import getPointCloud
from ma_sh.Method.render import renderGeometries
from ma_sh.Model.mash import Mash


def test():
    HOME = os.environ["HOME"]

    dataset_folder_path = HOME + "/chLi/Dataset/"

    mesh_folder_path = dataset_folder_path + "Mash/ShapeNet/normalized_mesh/"
    mash_folder_path = dataset_folder_path + "Mash/ShapeNet/normalized_mash/"
    sdf_folder_path = dataset_folder_path + "SDF-Coarse/ShapeNet/sdf/"

    classname_list = os.listdir(mesh_folder_path)

    error_file_path = HOME + '/chLi/Dataset/error.txt'

    solved_shape_num = 0
    for classname in classname_list:
        class_folder_path = mesh_folder_path + classname + "/"

        modelid_list = os.listdir(class_folder_path)

        for modelid in modelid_list:
            mesh_file_path = (
                class_folder_path + modelid + "/models/model_normalized.obj"
            )
            mash_file_path = (
                mash_folder_path
                + classname
                + "/"
                + modelid
                + "/models/model_normalized_obj.npy"
            )
            sdf_file_path = (
                sdf_folder_path
                + classname
                + "/"
                + modelid
                + "/models/model_normalized_obj.npy"
            )

            mesh = Mesh(mesh_file_path)
            mash = Mash.fromParamsFile(mash_file_path, device="cpu")
            sdf = np.load(sdf_file_path)

            mash_points = mash.toSamplePoints().numpy()
            sdf_points = sdf[sdf[:, 3] <= 0][:, :3]

            mesh_min_bound = np.min(mesh.vertices, axis=0)
            mesh_max_bound = np.max(mesh.vertices, axis=0)
            mesh_length = np.max(mesh_max_bound - mesh_min_bound)
            mesh_center = (mesh_min_bound + mesh_max_bound) / 2.0

            mesh_center_error = np.linalg.norm(mesh_center)
            mesh_length_error = np.abs(mesh_length - 0.9)

            mash_min_bound = np.min(mash_points, axis=0)
            mash_max_bound = np.max(mash_points, axis=0)
            mash_length = np.max(mash_max_bound - mash_min_bound)
            mash_center = (mash_min_bound + mash_max_bound) / 2.0

            mash_center_error = np.linalg.norm(mash_center)
            mash_length_error = np.abs(mash_length - 0.9)

            sdf_min_bound = np.min(sdf_points, axis=0)
            sdf_max_bound = np.max(sdf_points, axis=0)
            sdf_length = np.max(sdf_max_bound - sdf_min_bound)
            sdf_center = (sdf_min_bound + sdf_max_bound) / 2.0

            sdf_center_error = np.linalg.norm(sdf_center)
            sdf_length_error = np.abs(sdf_length - 0.9)

            error_info = ''
            if mesh_center_error > 1e-6 or mesh_length_error > 1e-6:
                print("current: class:", classname, "model:", modelid)
                print("mesh error!")
                print("mesh:", mesh_center_error, mesh_length_error)
                error_info += 'class:' + classname + ';model:' + modelid + ';mesh\n'
            if mash_center_error > 1e-2 or mash_length_error > 1e-2:
                print("current: class:", classname, "model:", modelid)
                print("mash error!")
                print("mash:", mash_center_error, mash_length_error)
                error_info += 'class:' + classname + ';model:' + modelid + ';mash\n'
            if sdf_center_error > 1e-2 or sdf_length_error > 1e-2:
                print("current: class:", classname, "model:", modelid)
                print("sdf error!")
                print("sdf:", sdf_center_error, sdf_length_error)
                error_info += 'class:' + classname + ';model:' + modelid + ';sdf\n'

            if error_info != '':
                with open(error_file_path, 'w+') as f:
                    f.write(error_info)

            if False:
                mash_pcd = getPointCloud(mash_points)
                sdf_pcd = getPointCloud(sdf_points)
                renderGeometries([mesh.toO3DMesh(), mash_pcd, sdf_pcd])
                exit()

            solved_shape_num += 1
            print('solved_shape_num:', solved_shape_num)

    exit()
    return True
