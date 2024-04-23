import os
import numpy as np

from ma_sh.Data.mesh import Mesh
from ma_sh.Method.path import createFileFolder


class Convertor(object):
    def __init__(
        self,
        shape_root_folder_path: str,
        dataset_root_folder_path: str,
        force_start: bool = False,
    ) -> None:
        self.shape_root_folder_path = shape_root_folder_path
        self.dataset_root_folder_path = dataset_root_folder_path
        self.force_start = force_start

        self.normalized_mesh_folder_path = (
            self.dataset_root_folder_path + "NormalizedMesh/"
        )
        self.tag_folder_path = self.dataset_root_folder_path + "Tag/NormalizedMesh/"
        return

    def convertOneShape(
        self, dataset_name: str, class_name: str, model_id: str
    ) -> bool:
        rel_file_path = dataset_name + "/" + class_name + "/" + model_id

        shape_file_path = (
            self.shape_root_folder_path
            + class_name
            + "/"
            + model_id
            + "/models/model_normalized.obj"
        )

        if not os.path.exists(shape_file_path):
            print("[ERROR][Convertor::convertOneShape]")
            print("\t shape file not exist!")
            print("\t shape_file_path:", shape_file_path)
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

        normalized_mesh_file_path = (
            self.normalized_mesh_folder_path + rel_file_path + ".obj"
        )

        createFileFolder(normalized_mesh_file_path)

        mesh = Mesh(shape_file_path)

        if not mesh.isValid():
            print("[ERROR][Convertor::convertOneShape]")
            print("\t mesh is not valid!")
            print("\t shape_file_path:", shape_file_path)
            return False

        min_bound = np.min(mesh.vertices, axis=0)
        max_bound = np.max(mesh.vertices, axis=0)
        length = np.max(max_bound - min_bound)
        scale = 0.9 / length
        center = (min_bound + max_bound) / 2.0

        mesh.vertices = (mesh.vertices - center) * scale

        mesh.save(normalized_mesh_file_path, True)

        with open(finish_tag_file_path, "w") as f:
            f.write("\n")

        return True

    def convertAll(self) -> bool:
        print("[INFO][Convertor::convertAll]")
        print("\t start convert all shapes to mashes...")
        solved_shape_num = 0

        classname_list = os.listdir(self.shape_root_folder_path)
        for classname in classname_list:
            class_folder_path = self.shape_root_folder_path + classname + "/"

            modelid_list = os.listdir(class_folder_path)

            for modelid in modelid_list:
                mesh_file_path = (
                    class_folder_path + modelid + "/models/model_normalized.obj"
                )

                if not os.path.exists(mesh_file_path):
                    continue

                self.convertOneShape("ShapeNet", classname, modelid)

                solved_shape_num += 1
                print("solved shape num:", solved_shape_num)

        return True
