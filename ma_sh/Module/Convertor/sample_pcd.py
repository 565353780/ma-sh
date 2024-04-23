import os
import numpy as np

from ma_sh.Data.mesh import Mesh
from ma_sh.Method.path import createFileFolder


class Convertor(object):
    def __init__(
        self,
        dataset_root_folder_path: str,
        gt_points_num: int = 400000,
        force_start: bool = False,
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path
        self.gt_points_num = gt_points_num
        self.force_start = force_start

        self.normalized_mesh_folder_path = (
            self.dataset_root_folder_path + "NormalizedMesh/"
        )
        self.sampled_pcd_folder_path = self.dataset_root_folder_path + "SampledPcd/"
        self.tag_folder_path = self.dataset_root_folder_path + "Tag/SampledPcd/"
        return

    def convertOneShape(
        self, dataset_name: str, class_name: str, model_id: str
    ) -> bool:
        rel_file_path = dataset_name + "/" + class_name + "/" + model_id

        normalized_mesh_file_path = (
            self.normalized_mesh_folder_path + rel_file_path + ".obj"
        )

        if not os.path.exists(normalized_mesh_file_path):
            print("[ERROR][Convertor::convertOneShape]")
            print("\t shape file not exist!")
            print("\t normalized_mesh_file_path:", normalized_mesh_file_path)
            return False

        finish_tag_file_path = self.tag_folder_path + rel_file_path + "/finish.txt"

        if os.path.exists(finish_tag_file_path):
            return True

        start_tag_file_path = self.tag_folder_path + rel_file_path + "/start.txt"

        if os.path.exists(start_tag_file_path):
            if not self.force_start:
                return True

        createFileFolder(start_tag_file_path)

        with open(start_tag_file_path, "w") as f:
            f.write("\n")

        sampled_pcd_file_path = self.sampled_pcd_folder_path + rel_file_path + ".npy"

        createFileFolder(sampled_pcd_file_path)

        mesh = Mesh(normalized_mesh_file_path)

        if not mesh.isValid():
            print("[ERROR][Convertor::convertOneShape]")
            print("\t mesh is not valid!")
            print("\t normalized_mesh_file_path:", normalized_mesh_file_path)
            return False

        try:
            points = mesh.toSamplePoints(self.gt_points_num)
        except:
            print("[ERROR][Convertor::convertOneShape]")
            print("\t toSamplePoints failed!")
            print("\t normalized_mesh_file_path:", normalized_mesh_file_path)
            return False

        if points is None:
            print("[ERROR][Convertor::convertOneShape]")
            print("\t toSamplePoints failed!")
            print("\t normalized_mesh_file_path:", normalized_mesh_file_path)
            return False

        np.save(sampled_pcd_file_path, points)

        with open(finish_tag_file_path, "w") as f:
            f.write("\n")

        return True

    def convertAll(self) -> bool:
        print("[INFO][Convertor::convertAll]")
        print("\t start convert all shapes to mashes...")
        solved_shape_num = 0

        dataset_folder_path = self.normalized_mesh_folder_path + "ShapeNet/"

        classname_list = os.listdir(dataset_folder_path)
        for classname in classname_list:
            class_folder_path = dataset_folder_path + classname + "/"

            modelid_list = os.listdir(class_folder_path)

            for model_file_name in modelid_list:
                modelid = model_file_name.split(".obj")[0]

                self.convertOneShape("ShapeNet", classname, modelid)

                solved_shape_num += 1
                print("solved shape num:", solved_shape_num)
        return True
