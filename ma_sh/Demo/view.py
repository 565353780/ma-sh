import os
import functools
import open3d as o3d

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

    if False:
        mash_params_file_path = "/Users/fufu/Downloads/model_normalized_obj.npy"

        mash = Mash.fromParamsFile(mash_params_file_path, 10, 10000, 0.4, device="cpu")

        mash.renderSamplePoints()

        points = mash.toSamplePoints().detach().clone().cpu().numpy()

        pcd = getPointCloud(points)

        o3d.io.write_point_cloud("./output/test.ply", pcd)

    if True:
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

    if False:
        o3d_viewer = O3DViewer()
        o3d_viewer.createWindow()

        mash_folder_path = "./output/20240421_00:51:09/"

        mash_filename_list = os.listdir(mash_folder_path)
        mash_filename_list.sort(key=functools.cmp_to_key(compare))

        for i, mash_filename in enumerate(mash_filename_list):
            if i != len(mash_filename_list) - 1:
                if i % 1000 != 0:
                    continue

            mash_file_path = mash_folder_path + mash_filename

            mash = Mash.fromParamsFile(mash_file_path, 10, 10000, 0.4, device="cpu")

            points = mash.toSamplePoints().detach().clone().cpu().numpy()

            pcd = getPointCloud(points)

            o3d_viewer.clearGeometries()
            o3d_viewer.addGeometry(pcd)

            print("now render is", i)

            o3d_viewer.update()

        o3d_viewer.run()
    return True
