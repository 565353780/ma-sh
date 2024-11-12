import os
import numpy as np

from ma_sh.Data.mesh import Mesh
from ma_sh.Method.path import createFileFolder
from ulip_manage.Module.detector import Detector

class Convertor(object):
    def __init__(
        self,
        dataset_root_folder_path: str,
        sample_point_num_list: list,
        device: str = "cuda:0",
        force_start: bool = False,
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path
        self.sample_point_num_list = sample_point_num_list
        self.device = device
        self.force_start = force_start

        self.mesh_folder_path = (
            self.dataset_root_folder_path + "ManifoldMesh/"
        )
        self.points_embedding_folder_path = (
            self.dataset_root_folder_path + "PointsEmbedding/"
        )
        self.tag_folder_path = self.dataset_root_folder_path + "Tag/PointsEmbedding/"

        model_file_path = '/home/chli/chLi/Model/ULIP2/pretrained_models_ckpt_zero-sho_classification_pointbert_ULIP-2.pt'
        open_clip_model_file_path = '/home/chli/Model/CLIP-ViT-bigG-14-laion2B-39B-b160k/open_clip_pytorch_model.bin'
        self.detector = Detector(model_file_path, open_clip_model_file_path, device)
        return

    def convertOneShape(
        self, dataset_name: str, class_name: str, model_id: str
    ) -> bool:
        rel_file_path = dataset_name + "/" + class_name + "/" + model_id

        mesh_file_path = self.mesh_folder_path + rel_file_path + ".obj"

        if not os.path.exists(mesh_file_path):
            print("[ERROR][Convertor::convertOneShape]")
            print("\t mesh file not exist!")
            print("\t mesh_file_path:", mesh_file_path)
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

        points_embedding_file_path = (
            self.points_embedding_folder_path + rel_file_path + ".npy"
        )

        createFileFolder(points_embedding_file_path)

        points_embedding_dict = {}

        mesh = Mesh(mesh_file_path)
        if not mesh.isValid():
            print("[WARN][Convertor::convertOneShape]")
            print("\t no valid mesh!")
            print("\t mesh_file_path:", mesh_file_path)
            return True

        for sample_point_num in self.sample_point_num_list:
            points = mesh.toSamplePoints(sample_point_num)

            points_embedding = self.detector.encodePointCloud(points).unsqueeze(0).cpu().numpy()

            points_embedding_dict[str(sample_point_num)] = points_embedding

        if len(points_embedding_dict.keys()) == 0:
            print("[WARN][Convertor::convertOneShape]")
            print("\t no valid sample num found!")
            print("\t sample_point_num_list:", self.sample_point_num_list)
            return True

        np.save(points_embedding_file_path, points_embedding_dict)

        with open(finish_tag_file_path, "w") as f:
            f.write("\n")

        return True

    def convertAll(self) -> bool:
        print("[INFO][Convertor::convertAll]")
        print("\t start convert all shapes to mashes...")
        solved_shape_num = 0

        dataset_folder_path = self.mesh_folder_path + "ShapeNet/"

        skip_file_list = [
            ['02691156', '157936971ef9b6bb858b20d410ebdb99'],
            ['02747177', '4c89f52fde77d653f4fb4dee5181bee'],
            ['02843684', '689f228f64564e663599338e3538d2bd'],
            ['02871439', 'ac0710b8484a44fe20dd2dd4d7d7656c'],
            ['02871439', 'ca56637ae250ff1259adebe36b392f27'],
            ['02992529', 'cf0dc63412541764cf9d394118de0d0'],
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
            ['04530566', '14fff3ebab1e144d6b77971fd22cc80d'],
        ]

        classname_list = os.listdir(dataset_folder_path)
        classname_list.sort()
        for classname in classname_list:
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
