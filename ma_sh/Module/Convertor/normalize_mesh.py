import os
import numpy as np
import torch

from ma_sh.Data.mesh import Mesh
from ma_sh.Method.path import createFileFolder
from ma_sh.Model.mash import Mash


class Convertor(object):
    def __init__(
        self,
        shape_root_folder_path: str,
        save_root_folder_path: str,
        force_start: bool = False,
    ) -> None:
        self.shape_root_folder_path = shape_root_folder_path
        self.save_root_folder_path = save_root_folder_path
        self.force_start = force_start
        return

    def convertOneShape(self, rel_shape_file_path: str) -> bool:
        shape_file_name = rel_shape_file_path.split("/")[-1]

        rel_shape_folder_path = rel_shape_file_path.split(shape_file_name)[0]

        shape_file_path = self.shape_root_folder_path + rel_shape_file_path

        if not os.path.exists(shape_file_path):
            print("[ERROR][Convertor::convertOneShape]")
            print("\t shape file not exist!")
            print("\t shape_file_path:", shape_file_path)
            return False

        unit_rel_folder_path = rel_shape_folder_path + shape_file_name.split(".")[0]

        finish_tag_file_path = (
            self.save_root_folder_path
            + "tag_normalize/"
            + unit_rel_folder_path
            + "/finish.txt"
        )

        if os.path.exists(finish_tag_file_path):
            return True

        start_tag_file_path = (
            self.save_root_folder_path
            + "tag_normalize/"
            + unit_rel_folder_path
            + "/start.txt"
        )

        if os.path.exists(start_tag_file_path):
            if not self.force_start:
                return True

        createFileFolder(start_tag_file_path)

        with open(start_tag_file_path, "w") as f:
            f.write("\n")

        save_mesh_file_path = (
            self.save_root_folder_path
            + "normalized_mesh/"
            + unit_rel_folder_path.replace("_obj", ".obj")
        )

        createFileFolder(save_mesh_file_path)

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
        move_vector = -scale * center

        mesh.vertices = mesh.vertices * scale + move_vector

        if not os.path.exists(save_mesh_file_path):
            mesh.save(save_mesh_file_path)

        save_pcd_file_path = (
            self.save_root_folder_path + "pcd/" + unit_rel_folder_path + ".npy"
        )

        save_normalize_pcd_file_path = (
            self.save_root_folder_path
            + "normalized_pcd/"
            + unit_rel_folder_path
            + ".npy"
        )

        if not os.path.exists(save_normalize_pcd_file_path):
            if os.path.exists(save_pcd_file_path):
                points = np.load(save_pcd_file_path)
                points = points * scale + move_vector

                createFileFolder(save_normalize_pcd_file_path)
                np.save(save_normalize_pcd_file_path, points)

        save_mash_file_path = (
            self.save_root_folder_path + "mash/" + unit_rel_folder_path + ".npy"
        )

        save_normalize_mash_file_path = (
            self.save_root_folder_path
            + "normalized_mash/"
            + unit_rel_folder_path
            + ".npy"
        )

        if not os.path.exists(save_normalize_mash_file_path):
            if os.path.exists(save_mash_file_path):
                mash = Mash.fromParamsFile(save_mash_file_path, device="cpu")
                mash.positions = mash.positions * scale + torch.from_numpy(move_vector)
                mash.sh_params *= scale

                mash.saveParamsFile(save_normalize_mash_file_path, True)

        with open(finish_tag_file_path, "w") as f:
            f.write("\n")

        return True

    def convertAll(self) -> bool:
        os.makedirs(self.save_root_folder_path, exist_ok=True)

        print("[INFO][Convertor::convertAll]")
        print("\t start convert all shapes to mashes...")
        solved_shape_num = 0
        for root, _, files in os.walk(self.shape_root_folder_path):
            for filename in files:
                if filename[-4:] not in [".obj", ".ply"]:
                    continue


                rel_file_path = (
                    root.split(self.shape_root_folder_path)[1] + "/" + filename
                )

                self.convertOneShape(rel_file_path)

                solved_shape_num += 1
                print("solved shape num:", solved_shape_num)

        return True
