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

                self.convertOneShape("ShapeNet", classname, modelid)

                solved_shape_num += 1
                print("solved shape num:", solved_shape_num)
        return True
