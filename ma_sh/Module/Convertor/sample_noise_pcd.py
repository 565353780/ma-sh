import os
import numpy as np

from ma_sh.Method.path import createFileFolder


class Convertor(object):
    def __init__(
        self,
        dataset_root_folder_path: str,
        gt_points_num: int = 400000,
        noise_sigma: float = 0.01,
        force_start: bool = False,
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path
        self.gt_points_num = gt_points_num
        self.noise_sigma = noise_sigma
        self.force_start = force_start

        self.sampled_pcd_folder_path = (
            self.dataset_root_folder_path + "SampledPcd_Manifold/"
        )
        self.sampled_noise_pcd_folder_path = self.dataset_root_folder_path + "SampledPcd_Manifold_Noise_" + str(noise_sigma).replace('.', '-') + "/"
        self.tag_folder_path = self.dataset_root_folder_path + "Tag/SampledPcd_Manifold_Noise_" + str(noise_sigma).replace('.', '-') + "/"
        return

    def convertOneShape(
        self, dataset_name: str, class_name: str, model_id: str
    ) -> bool:
        rel_file_path = dataset_name + "/" + class_name + "/" + model_id

        sampled_pcd_file_path = (
            self.sampled_pcd_folder_path + rel_file_path + ".npy"
        )

        if not os.path.exists(sampled_pcd_file_path):
            print("[ERROR][Convertor::convertOneShape]")
            print("\t shape file not exist!")
            print("\t sampled_pcd_file_path:", sampled_pcd_file_path)
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

        sampled_noise_pcd_file_path = self.sampled_noise_pcd_folder_path + rel_file_path + ".npy"

        createFileFolder(sampled_noise_pcd_file_path)

        points = np.load(sampled_pcd_file_path)

        gaussian_noise = np.random.normal(0, self.noise_sigma, points.shape)

        noise_points = points + gaussian_noise

        np.save(sampled_noise_pcd_file_path, noise_points)

        with open(finish_tag_file_path, "w") as f:
            f.write("\n")

        return True

    def convertAll(self) -> bool:
        print("[INFO][Convertor::convertAll]")
        print("\t start convert all shapes to mashes...")
        solved_shape_num = 0

        dataset_folder_path = self.sampled_pcd_folder_path + "ShapeNet/"

        classname_list = os.listdir(dataset_folder_path)
        classname_list.sort()
        first_solve_class = ['03001627', '02691156', '02958343']
        first_solve_models = [
            '3020942d1547cf562056b4bd5d870b47', '2acc2a87aef7cc559ca96b2737246fca',
            '2b90701386f1813052db1dda4adf0a0c', '28445d445cb8b3aa5de04aad18bd94c3',
            '357e2dd1512b96168e2b488ea5fa466a', '23babf1dd4209349def08b067722d9e5',
        ]
        for classname in classname_list:
            if classname != first_solve_class[0]:
                continue

            class_folder_path = dataset_folder_path + classname + "/"

            modelid_list = os.listdir(class_folder_path)
            modelid_list.sort()

            for model_file_name in modelid_list:
                modelid = model_file_name.split(".npy")[0]

                if modelid not in first_solve_models:
                    continue

                self.convertOneShape("ShapeNet", classname, modelid)

                solved_shape_num += 1
                print("solved shape num:", solved_shape_num)
        return True
