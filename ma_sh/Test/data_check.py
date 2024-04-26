import os
import numpy as np

from ma_sh.Data.mesh import Mesh
from ma_sh.Method.pcd import getPointCloud
from ma_sh.Method.path import createFileFolder
from ma_sh.Method.render import renderGeometries
from ma_sh.Model.mash import Mash


def view_data():
    HOME = os.environ["HOME"]

    dataset_folder_path = HOME + "/Dataset/"

    mesh_folder_path = HOME + "/chLi/Dataset/" + "NormalizedMesh/ShapeNet/"
    mash_folder_path = dataset_folder_path + "MashV2/ShapeNet/"
    sdf_folder_path = dataset_folder_path + "SampledSDF_0_025/ShapeNet/"

    classname_list = os.listdir(mash_folder_path)

    for classname in classname_list:
        if classname != "03001627":
            continue

        class_folder_path = mash_folder_path + classname + "/"

        modelid_list = os.listdir(class_folder_path)

        for modelname in modelid_list:
            modelid = modelname.split(".npy")[0]
            mash_file_path = class_folder_path + modelid + ".npy"
            mesh_file_path = mesh_folder_path + classname + "/" + modelid + ".obj"
            sdf_file_path = sdf_folder_path + classname + "/" + modelid + ".npy"

            render_list = []

            if True:
                mesh = Mesh(mesh_file_path)
                render_list.append(mesh.toO3DMesh())

            if False:
                mash = Mash.fromParamsFile(mash_file_path, device="cpu")
                mash_points = mash.toSamplePoints().numpy()

                mash_pcd = getPointCloud(mash_points)
                mash_pcd.translate([0, 1, 0])
                render_list.append(mash_pcd)

            if False:
                sdf = np.load(sdf_file_path)
                sdf_points = sdf[sdf[:, 3] <= 0][:, :3]

                sdf_pcd = getPointCloud(sdf_points)
                sdf_pcd.translate([0, -1, 0])
                render_list.append(sdf_pcd)

            print("current model:", classname, modelid)
            renderGeometries(render_list)

    exit()
    return True


def view_error_data():
    HOME = os.environ["HOME"]

    dataset_folder_path = HOME + "/Dataset/"

    mesh_folder_path = HOME + "/chLi/Dataset/" + "NormalizedMesh/ShapeNet/"
    mash_folder_path = dataset_folder_path + "MashV2/ShapeNet/"
    sdf_folder_path = dataset_folder_path + "SampledSDF_0_025/ShapeNet/"
    error_file_path = HOME + "/Dataset/error.txt"

    with open(error_file_path, "r") as f:
        error_data_list = f.readlines()

    for error_data in error_data_list:
        error_item_list = error_data.split("\n")[0].split(";")

        error_classname = error_item_list[0].split("class:")[1]
        error_modelid = error_item_list[1].split("model:")[1]

        mesh_file_path = (
            mesh_folder_path + error_classname + "/" + error_modelid + ".obj"
        )
        mash_file_path = (
            mash_folder_path + error_classname + "/" + error_modelid + ".npy"
        )
        sdf_file_path = sdf_folder_path + error_classname + "/" + error_modelid + ".npy"

        print(error_data)

        mesh = Mesh(mesh_file_path)

        mash = Mash.fromParamsFile(mash_file_path, device="cpu")
        mash_points = mash.toSamplePoints().numpy()

        sdf = np.load(sdf_file_path)
        sdf_points = sdf[sdf[:, 3] <= 0][:, :3]

        mash_pcd = getPointCloud(mash_points)
        sdf_pcd = getPointCloud(sdf_points)
        renderGeometries([mesh.toO3DMesh(), mash_pcd, sdf_pcd])

    exit()
    return True


def test():
    # view_data()
    # view_error_data()

    HOME = os.environ["HOME"]

    dataset_folder_path = HOME + "/Dataset/"

    mesh_folder_path = HOME + "/chLi/Dataset/" + "NormalizedMesh/ShapeNet/"
    mash_folder_path = dataset_folder_path + "MashV2/ShapeNet/"
    sdf_folder_path = dataset_folder_path + "SampledSDF_0_025/ShapeNet/"
    tag_folder_path = dataset_folder_path + "Tag/DataCheck/ShapeNet/"
    error_file_path = HOME + "/Dataset/error_0_025.txt"

    os.makedirs(tag_folder_path, exist_ok=True)

    classname_list = os.listdir(mash_folder_path)

    solved_shape_num = 0
    for classname in classname_list:
        class_folder_path = mash_folder_path + classname + "/"

        filename_list = os.listdir(class_folder_path)

        for filename in filename_list:
            modelid = filename.split(".obj")[0]

            mesh_file_path = mesh_folder_path + classname + "/" + modelid + ".obj"
            mash_file_path = mash_folder_path + classname + "/" + modelid + ".npy"
            sdf_file_path = sdf_folder_path + classname + "/" + modelid + ".npy"

            finish_tag_file_path = (
                tag_folder_path + classname + "/" + modelid + "_finish.txt"
            )

            if os.path.exists(finish_tag_file_path):
                solved_shape_num += 1
                print("solved_shape_num:", solved_shape_num)
                continue

            if os.path.exists(mesh_file_path):
                mesh = Mesh(mesh_file_path)

                mesh_min_bound = np.min(mesh.vertices, axis=0)
                mesh_max_bound = np.max(mesh.vertices, axis=0)
                mesh_length = np.max(mesh_max_bound - mesh_min_bound)
                mesh_center = (mesh_min_bound + mesh_max_bound) / 2.0

                mesh_center_error = np.linalg.norm(mesh_center)
                mesh_length_error = np.abs(mesh_length - 0.9)

                if mesh_center_error > 1e-6 or mesh_length_error > 1e-6:
                    print("current: class:", classname, "model:", modelid)
                    print("mesh error!")
                    print("mesh:", mesh_center_error, mesh_length_error)
                    error_info = "class:" + classname + ";model:" + modelid + ";mesh\n"

                    with open(error_file_path, "a+") as f:
                        f.write(error_info)

            if os.path.exists(mash_file_path):
                mash = Mash.fromParamsFile(mash_file_path, device="cpu")
                mash_points = mash.toSamplePoints().numpy()

                mash_min_bound = np.min(mash_points, axis=0)
                mash_max_bound = np.max(mash_points, axis=0)
                mash_length = np.max(mash_max_bound - mash_min_bound)
                mash_center = (mash_min_bound + mash_max_bound) / 2.0

                mash_center_error = np.linalg.norm(mash_center)
                mash_length_error = np.abs(mash_length - 0.9)

                if mash_center_error > 1e-2 or mash_length_error > 1e-2:
                    print("current: class:", classname, "model:", modelid)
                    print("mash error!")
                    print("mash:", mash_center_error, mash_length_error)
                    error_info = "class:" + classname + ";model:" + modelid + ";mash\n"

                    with open(error_file_path, "a+") as f:
                        f.write(error_info)

            if os.path.exists(sdf_file_path):
                sdf = np.load(sdf_file_path)
                sdf_points = sdf[sdf[:, 3] <= 0][:, :3]

                sdf_min_bound = np.min(sdf_points, axis=0)
                sdf_max_bound = np.max(sdf_points, axis=0)
                sdf_length = np.max(sdf_max_bound - sdf_min_bound)
                sdf_center = (sdf_min_bound + sdf_max_bound) / 2.0

                sdf_center_error = np.linalg.norm(sdf_center)
                sdf_length_error = np.abs(sdf_length - 0.9)

                if sdf_center_error > 1e-2 or sdf_length_error > 1e-2:
                    print("current: class:", classname, "model:", modelid)
                    print("sdf error!")
                    print("sdf:", sdf_center_error, sdf_length_error)
                    error_info = "class:" + classname + ";model:" + modelid + ";sdf\n"

                    with open(error_file_path, "a+") as f:
                        f.write(error_info)

            createFileFolder(finish_tag_file_path)
            with open(finish_tag_file_path, "w") as f:
                f.write("\n")
            solved_shape_num += 1
            print("solved_shape_num:", solved_shape_num)

    exit()
    return True
