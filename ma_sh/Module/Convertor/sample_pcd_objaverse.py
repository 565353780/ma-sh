import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

from ma_sh.Data.mesh import Mesh
from ma_sh.Method.path import createFileFolder, removeFile


class Convertor(object):
    def __init__(
        self,
        dataset_root_folder_path: str,
        dataset_name: str,
        gt_points_num: int = 400000,
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path
        self.gt_points_num = gt_points_num

        self.manifold_mesh_folder_path = (
            self.dataset_root_folder_path + dataset_name + "/manifold/"
        )
        self.sampled_pcd_folder_path = self.dataset_root_folder_path + dataset_name + "/manifold_pcd/"
        return

    def convertOneShape(
        self, model_id: str
    ) -> bool:
        rel_file_path = model_id

        sampled_pcd_file_path = self.sampled_pcd_folder_path + rel_file_path + ".npy"

        if os.path.exists(sampled_pcd_file_path):
            return True

        manifold_mesh_file_path = (
            self.manifold_mesh_folder_path + rel_file_path + ".obj"
        )

        if not os.path.exists(manifold_mesh_file_path):
            print("[ERROR][Convertor::convertOneShape]")
            print("\t shape file not exist!")
            print("\t manifold_mesh_file_path:", manifold_mesh_file_path)
            return False

        start_tag_file_path = self.sampled_pcd_folder_path + rel_file_path + "_start.txt"

        if os.path.exists(start_tag_file_path):
            return True

        createFileFolder(start_tag_file_path)

        with open(start_tag_file_path, "w") as f:
            f.write("\n")

        sampled_pcd_file_path = self.sampled_pcd_folder_path + rel_file_path + ".npy"

        createFileFolder(sampled_pcd_file_path)

        mesh = Mesh(manifold_mesh_file_path)

        if not mesh.isValid():
            print("[ERROR][Convertor::convertOneShape]")
            print("\t mesh is not valid!")
            print("\t manifold_mesh_file_path:", manifold_mesh_file_path)
            return False

        try:
            points = mesh.toSamplePoints(self.gt_points_num)
        except:
            print("[ERROR][Convertor::convertOneShape]")
            print("\t toSamplePoints failed!")
            print("\t manifold_mesh_file_path:", manifold_mesh_file_path)
            return False

        if points is None:
            print("[ERROR][Convertor::convertOneShape]")
            print("\t toSamplePoints failed!")
            print("\t manifold_mesh_file_path:", manifold_mesh_file_path)
            return False

        np.save(sampled_pcd_file_path, points)

        removeFile(start_tag_file_path)

        return True

    def convertAll(self, worker_num: int = 6) -> bool:
        print("[INFO][Convertor::convertAll]")
        print("\t start convert all shapes to mashes...")

        dataset_folder_path = self.manifold_mesh_folder_path

        classname_list = os.listdir(dataset_folder_path)
        classname_list.sort()

        for classname in classname_list:
            class_folder_path = dataset_folder_path + classname + "/"

            modelid_list = os.listdir(class_folder_path)
            modelid_list.sort()

            full_model_id_list = [classname + '/' + model_id[:-4] for model_id in modelid_list]

            print("[INFO][Convertor::convertAll]")
            print('\t start convert all objaverse meshes to sample pcd :', classname, '...')
            with Pool(worker_num) as pool:
                results = list(tqdm(
                    pool.imap(self.convertOneShape, full_model_id_list),
                    total=len(full_model_id_list),
                    desc="Processing"
                ))

        return True
