import os
import open3d as o3d

from ma_sh.Method.path import createFileFolder
from ma_sh.Model.mash import Mash

class Convertor(object):
    def __init__(
        self,
        dataset_root_folder_path: str,
        device: str = "cuda:0",
        force_start: bool = False,
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path
        self.device = device
        self.force_start = force_start

        current_appendix = ''
        self.mash_folder_path = (
            self.dataset_root_folder_path + "MashV4" + current_appendix + "/"
        )
        self.mash_pcd_folder_path = (
            self.dataset_root_folder_path + "MashPcd_Manifold" + current_appendix + "/"
        )
        self.tag_folder_path = self.dataset_root_folder_path + "Tag/MashPcd_Manifold" + current_appendix + "/"
        return

    def convertOneShape(
        self, dataset_name: str, class_name: str, model_id: str
    ) -> bool:
        rel_file_path = dataset_name + "/" + class_name + "/" + model_id

        mash_file_path = (
            self.mash_folder_path + rel_file_path + ".npy"
        )

        if not os.path.exists(mash_file_path):
            print("[ERROR][Convertor::convertOneShape]")
            print("\t shape folder not exist!")
            print("\t mash_file_path:", mash_file_path)
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

        mash_pcd_file_path = (
            self.mash_pcd_folder_path + rel_file_path + ".ply"
        )

        createFileFolder(mash_pcd_file_path)

        mash = Mash.fromParamsFile(mash_file_path, 90, 1000, 0.8, device=self.device)
        mash_pcd = mash.toSamplePcd(False, False, 0.1)
        o3d.io.write_point_cloud(mash_pcd_file_path, mash_pcd, write_ascii=True)

        with open(finish_tag_file_path, "w") as f:
            f.write("\n")

        return True

    def convertAll(self) -> bool:
        print("[INFO][Convertor::convertAll]")
        print("\t start convert all shapes to mashes...")
        solved_shape_num = 0

        dataset_folder_path = self.mash_folder_path + "ShapeNet/"

        classname_list = os.listdir(dataset_folder_path)
        classname_list.sort()
        first_solve_class = ['03001627', '02691156', '02958343']
        for classname in classname_list:
            if classname != first_solve_class[1]:
                continue

            class_folder_path = dataset_folder_path + classname + "/"

            modelid_list = os.listdir(class_folder_path)
            modelid_list.sort()

            for model_file_name in modelid_list:
                modelid = model_file_name.split(".npy")[0]

                self.convertOneShape("ShapeNet", classname, modelid)

                solved_shape_num += 1
                print("solved shape num:", solved_shape_num)
        return True
