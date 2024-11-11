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

        skip_file_list = [
            ['02691156', '157936971ef9b6bb858b20d410ebdb99'],
            ['02747177', '4c89f52fde77d653f4fb4dee5181bee'],
            ['02871439', 'ac0710b8484a44fe20dd2dd4d7d7656c'],
            ['02871439', 'ca56637ae250ff1259adebe36b392f27'],
            ['02992529', 'cf0dc63412541764cf9d394118de0d0'],
            ['03085013', 'ef3d038046cab5cabeb3159acb187cec'],
            ['03211117', 'c08e59a3d09cbabf6002a1da9aad9f4f'],
            ['03211117', 'd8f4c5160059ef245d79a44cb814180d'],
            ['03211117', 'ecd1641932584115fcea08a6f6e1c30a'],
            ['03325088', '8b96750136eac3c5c36fb70296e45483'],
            ['03325088', 'cbac6ded2160f1beb362845c6edb57fc'],
            ['03325088', 'efaff4f34573d23caac6c989b01c7'],
            ['03325088', 'f9ff34cbd52103d3d42b9650f19dd425'],
            ['04090263', 'cb5e01162787772ff7bd077790d66b82'],
            ['04256520', '8d5acb33654685d965715e89ab65beed'],
            ['04379243', '2783c8d705a1a146668ae11a7db5e82a'],
            ['04379243', '8a91b91802db34ea409421506a05b6e1'],
            ['04379243', '9affa2569ec8968c60edf8bc2f5c8881'],
            ['04379243', 'ba2f81e15029a37baf7caa8fd318856'],
            ['04379243', 'cafca523ae3653502454f22008de5a3e'],
            ['04379243', 'db94dde04aad570d2f8bc0d6e7c6775'],
        ]

        classname_list = os.listdir(dataset_folder_path)
        classname_list.sort()
        for classname in classname_list:
            if '.zip' in classname:
                continue

            class_folder_path = dataset_folder_path + classname + "/"

            modelid_list = os.listdir(class_folder_path)
            modelid_list.sort()

            for model_file_name in modelid_list:
                if solved_shape_num < 0:
                    solved_shape_num += 1
                    continue

                modelid = model_file_name.split(".obj")[0]

                if [classname, modelid] in skip_file_list:
                    solved_shape_num += 1
                    print("solved shape num:", solved_shape_num)
                    continue

                print('[\'' + classname + '\', \'' + modelid + '\'],')

                self.convertOneShape("ShapeNet", classname, modelid)

                solved_shape_num += 1
                print("solved shape num:", solved_shape_num)
        return True
