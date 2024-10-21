import os

from sdf_generate.Method.sample_sdf import convertSDFNearSurface

from ma_sh.Method.path import createFileFolder


class Convertor(object):
    def __init__(
        self,
        dataset_root_folder_path: str,
        sample_sdf_point_num: int = 250000,
        gauss_noise: float = 0.0025,
        force_start: bool = False,
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path
        self.sample_sdf_point_num = sample_sdf_point_num
        self.gauss_noise = gauss_noise
        self.force_start = force_start

        sdf_label = "SampledSDF_" + str(gauss_noise).replace(".", "_")

        self.manifold_mesh_folder_path = self.dataset_root_folder_path + "ManifoldMesh/"
        self.sampled_sdf_folder_path = self.dataset_root_folder_path + sdf_label + "/"
        self.tag_folder_path = self.dataset_root_folder_path + "Tag/" + sdf_label + "/"
        return

    def convertOneShape(
        self, dataset_name: str, class_name: str, model_id: str
    ) -> bool:
        rel_file_path = dataset_name + "/" + class_name + "/" + model_id

        manifold_mesh_file_path = (
            self.manifold_mesh_folder_path + rel_file_path + ".obj"
        )

        if not os.path.exists(manifold_mesh_file_path):
            print("[ERROR][Convertor::convertOneShape]")
            print("\t shape file not exist!")
            print("\t manifold_mesh_file_path:", manifold_mesh_file_path)
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

        sampled_sdf_file_path = self.sampled_sdf_folder_path + rel_file_path + ".npy"

        createFileFolder(sampled_sdf_file_path)

        try:
            convertSDFNearSurface(
                manifold_mesh_file_path,
                sampled_sdf_file_path,
                self.sample_sdf_point_num,
                self.gauss_noise,
                True,
            )
        except:
            print("[ERROR][Convertor::convertOneShape]")
            print("\t convertSDFNearSurface failed!")
            print("\t manifold_mesh_file_path:", manifold_mesh_file_path)
            return False

        with open(finish_tag_file_path, "w") as f:
            f.write("\n")

        return True

    def convertAll(self) -> bool:
        print("[INFO][Convertor::convertAll]")
        print("\t start convert all shapes to mashes...")
        solved_shape_num = 0

        dataset_folder_path = self.manifold_mesh_folder_path + "ShapeNet/"

        classname_list = os.listdir(dataset_folder_path)
        classname_list.sort()
        for classname in classname_list:
            class_folder_path = dataset_folder_path + classname + "/"

            modelid_list = os.listdir(class_folder_path)
            modelid_list.sort()

            for model_file_name in modelid_list:
                modelid = model_file_name.split(".obj")[0]

                self.convertOneShape("ShapeNet", classname, modelid)

                solved_shape_num += 1
                print("solved shape num:", solved_shape_num)
        return True
