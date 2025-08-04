import torch

from ma_sh.Model.mash import Mash
from data_convert.Module.base_convertor import BaseConvertor


class Convertor(BaseConvertor):
    def __init__(
        self,
        source_root_folder_path: str,
        target_root_folder_path: str,
        sample_phi_num: int = 40,
        sample_theta_num: int = 40,
        points_per_submesh: int = 1024,
        dtype=torch.float32,
        device: str = "cuda",
    ) -> None:
        super().__init__(source_root_folder_path, target_root_folder_path)

        self.sample_phi_num = sample_phi_num
        self.sample_theta_num = sample_theta_num
        self.points_per_submesh = points_per_submesh
        self.dtype = dtype
        self.device = device
        return

    def convertData(self, source_path: str, target_path: str) -> bool:
        mash = Mash.fromParamsFile(
            source_path,
            self.sample_phi_num,
            self.sample_theta_num,
            self.dtype,
            self.device,
        )

        mash.saveAsPcdFile(target_path)
        return True
