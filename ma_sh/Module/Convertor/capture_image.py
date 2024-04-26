import os

from open3d_manage.Module.shape_image_sampler import ShapeImageSampler

from ma_sh.Method.path import createFileFolder


class Convertor(object):
    def __init__(
        self,
        dataset_root_folder_path: str,
        window_name: str = "Open3D",
        width: int = 1920,
        height: int = 1080,
        left: int = 50,
        top: int = 50,
        visible: bool = True,
        y_rotate_num: int = 8,
        x_rotate_num: int = 5,
        x_save_idxs: list = [1, 2, 3],
        force_start: bool = False,
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path
        self.y_rotate_num = y_rotate_num
        self.x_rotate_num = x_rotate_num
        self.x_save_idxs = x_save_idxs
        self.force_start = force_start

        self.shape_image_sampler = ShapeImageSampler(
            window_name, width, height, left, top, visible
        )

        self.manifold_mesh_folder_path = self.dataset_root_folder_path + "ManifoldMesh/"
        self.captured_image_folder_path = (
            self.dataset_root_folder_path + "CapturedImage/"
        )
        self.tag_folder_path = self.dataset_root_folder_path + "Tag/CapturedImage/"
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

        captured_image_folder_path = (
            self.captured_image_folder_path + rel_file_path + "/"
        )

        os.makedirs(captured_image_folder_path, exist_ok=True)

        self.shape_image_sampler.sampleImages(
            manifold_mesh_file_path,
            captured_image_folder_path,
            self.y_rotate_num,
            self.x_rotate_num,
            self.x_save_idxs,
            True,
        )

        with open(finish_tag_file_path, "w") as f:
            f.write("\n")

        return True

    def convertAll(self) -> bool:
        print("[INFO][Convertor::convertAll]")
        print("\t start convert all shapes to mashes...")
        solved_shape_num = 0

        dataset_folder_path = self.manifold_mesh_folder_path + "ShapeNet/"

        classname_list = os.listdir(dataset_folder_path)
        for classname in classname_list:
            if classname != "03001627":
                continue

            class_folder_path = dataset_folder_path + classname + "/"

            modelid_list = os.listdir(class_folder_path)

            for model_file_name in modelid_list:
                modelid = model_file_name.split(".obj")[0]

                self.convertOneShape("ShapeNet", classname, modelid)

                solved_shape_num += 1
                print("solved shape num:", solved_shape_num)
        return True
