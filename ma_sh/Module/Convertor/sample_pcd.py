import os
import numpy as np

from ma_sh.Data.mesh import Mesh
from ma_sh.Method.path import createFileFolder


class Convertor(object):
    def __init__(
        self,
        shape_root_folder_path: str,
        save_root_folder_path: str,
        force_start: bool = False,
        gt_points_num: int = 400000,
    ) -> None:
        self.shape_root_folder_path = shape_root_folder_path
        self.save_root_folder_path = save_root_folder_path
        self.force_start = force_start
        self.gt_points_num = gt_points_num
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
            + "tag_pcd/"
            + unit_rel_folder_path
            + "/finish.txt"
        )

        if os.path.exists(finish_tag_file_path):
            return True

        start_tag_file_path = (
            self.save_root_folder_path
            + "tag_pcd/"
            + unit_rel_folder_path
            + "/start.txt"
        )

        save_pcd_file_path = (
            self.save_root_folder_path + "pcd/" + unit_rel_folder_path + ".npy"
        )

        if os.path.exists(start_tag_file_path):
            if not self.force_start:
                return True

        createFileFolder(start_tag_file_path)

        with open(start_tag_file_path, "w") as f:
            f.write("\n")

        save_pcd_file_path = (
            self.save_root_folder_path + "pcd/" + unit_rel_folder_path + ".npy"
        )

        if os.path.exists(save_pcd_file_path):
            with open(finish_tag_file_path, "w") as f:
                f.write("\n")
            return True

        createFileFolder(save_pcd_file_path)

        mesh = Mesh(shape_file_path)

        if not mesh.isValid():
            print("[ERROR][Convertor::convertOneShape]")
            print("\t mesh is not valid!")
            print("\t shape_file_path:", shape_file_path)
            return False

        points = mesh.toSamplePoints(self.gt_points_num)

        if points is None:
            print("[ERROR][Convertor::convertOneShape]")
            print("\t toSamplePoints failed!")
            print("\t shape_file_path:", shape_file_path)
            return False

        np.save(save_pcd_file_path, points)

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
