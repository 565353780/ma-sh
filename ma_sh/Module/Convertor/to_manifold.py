import os
import open3d as o3d

from sdf_generate.Method.to_manifold import toManifold

from ma_sh.Method.path import createFileFolder, removeFile


class Convertor(object):
    def __init__(
        self,
        dataset_root_folder_path: str,
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path

        self.normalized_mesh_folder_path = (
            self.dataset_root_folder_path + "Objaverse_82K/mesh/"
        )
        self.manifold_mesh_folder_path = self.dataset_root_folder_path + "Objaverse_82K/manifold/"
        return

    def convertOneShape(
        self, model_id: str
    ) -> bool:
        rel_file_path = model_id

        manifold_mesh_file_path = (
            self.manifold_mesh_folder_path + rel_file_path + ".obj"
        )

        if os.path.exists(manifold_mesh_file_path):
            return True

        ply_mesh_file_path = (
            self.normalized_mesh_folder_path + rel_file_path + ".ply"
        )

        normalized_mesh_file_path = (
            self.normalized_mesh_folder_path + rel_file_path + ".obj"
        )

        if not os.path.exists(normalized_mesh_file_path):
            if not os.path.exists(ply_mesh_file_path):
                print("[ERROR][Convertor::convertOneShape]")
                print("\t shape file not exist!")
                print("\t normalized_mesh_file_path:", normalized_mesh_file_path)
                return False

            mesh = o3d.io.read_triangle_mesh(ply_mesh_file_path)
            o3d.io.write_triangle_mesh(normalized_mesh_file_path, mesh, write_ascii=True)

        if os.path.exists(ply_mesh_file_path):
            removeFile(ply_mesh_file_path)

        start_tag_file_path = (
            self.manifold_mesh_folder_path + rel_file_path + "_start.txt"
        )

        if os.path.exists(start_tag_file_path):
            return True

        createFileFolder(start_tag_file_path)

        with open(start_tag_file_path, "w") as f:
            f.write("\n")

        createFileFolder(manifold_mesh_file_path)

        try:
            toManifold(normalized_mesh_file_path, manifold_mesh_file_path, False)
        except:
            print("[ERROR][Convertor::convertOneShape]")
            print("\t toManifold failed!")
            print("\t normalized_mesh_file_path:", normalized_mesh_file_path)
            return False

        removeFile(start_tag_file_path)

        return True

    def convertAll(self) -> bool:
        print("[INFO][Convertor::convertAll]")
        print("\t start convert all shapes to mashes...")
        solved_shape_num = 0

        dataset_folder_path = self.normalized_mesh_folder_path

        classname_list = os.listdir(dataset_folder_path)
        classname_list.sort()
        for classname in classname_list:
            class_folder_path = dataset_folder_path + classname + "/"

            modelid_list = os.listdir(class_folder_path)
            modelid_list.sort()

            for modelid in modelid_list:
                if modelid.endswith('.txt'):
                    continue

                model_id = classname + '/' + modelid[:-4]

                self.convertOneShape(model_id)

                solved_shape_num += 1
                print("solved shape num:", solved_shape_num)
        return True
