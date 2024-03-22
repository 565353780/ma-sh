import os
import torch

from ma_sh.Model.mash import Mash


class PcdConvertor(object):
    def __init__(
        self,
        save_root_folder_path: str,
        mask_boundary_sample_num: int = 10,
        sample_polar_num: int = 10,
        sample_point_scale: float = 0.5,
        use_inv: bool = True,
        idx_dtype=torch.int64,
        dtype=torch.float64,
        device: str = "cpu",
    ) -> None:
        self.save_root_folder_path = save_root_folder_path
        self.mask_boundary_sample_num = mask_boundary_sample_num
        self.sample_polar_num = sample_polar_num
        self.sample_point_scale = sample_point_scale
        self.use_inv = use_inv
        self.idx_dtype = idx_dtype
        self.dtype = dtype
        self.device = device
        return

    def createMashFromFile(self, params_file_path: str):
        mash = Mash.fromParamsFile(
            params_file_path,
            self.mask_boundary_sample_num,
            self.sample_polar_num,
            self.sample_point_scale,
            self.idx_dtype,
            self.dtype,
            self.device,
        )

        return mash

    def convertOneShape(self, rel_params_file_path: str) -> bool:
        params_file_name = rel_params_file_path.split("/")[-1]

        rel_params_folder_path = rel_params_file_path.split(params_file_name)[0]

        params_file_path = self.save_root_folder_path + "mash/" + rel_params_file_path

        if not os.path.exists(params_file_path):
            print("[ERROR][PcdConvertor::convertOneShape]")
            print("\t params file not exist!")
            print("\t params_file_path:", params_file_path)
            return False

        unit_rel_file_path = rel_params_folder_path + params_file_name.replace(
            ".npy", ".ply"
        )

        pcd_file_path = self.save_root_folder_path + "pcd/" + unit_rel_file_path

        if os.path.exists(pcd_file_path):
            return True

        mash = self.createMashFromFile(params_file_path)

        mash.saveAsPcdFile(pcd_file_path)
        return True

    def convertAll(self) -> bool:
        os.makedirs(self.save_root_folder_path, exist_ok=True)

        print("[INFO][PcdConvertor::convertAll]")
        print("\t start convert all mash to pcds...")
        solved_shape_num = 0
        for root, _, files in os.walk(self.save_root_folder_path + "mash/"):
            for filename in files:
                if filename[-4:] != ".npy":
                    continue

                rel_file_path = (
                    root.split(self.save_root_folder_path + "mash/")[1] + "/" + filename
                )

                self.convertOneShape(rel_file_path)

                solved_shape_num += 1
                print("solved mash num:", solved_shape_num)

        return True
