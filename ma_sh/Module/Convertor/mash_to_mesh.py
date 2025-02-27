import sys
sys.path.append('../wn-nc')

import torch

from ma_sh.Model.mash import Mash
from ma_sh.Module.Convertor.base_convertor import BaseConvertor


class Convertor(BaseConvertor):
    def __init__(
        self,
        source_root_folder_path: str,
        target_root_folder_path: str,
    ) -> None:
        super().__init__(source_root_folder_path, target_root_folder_path)
        return

    def convertData(self, source_path: str, target_path: str) -> bool:
        mash = Mash.fromParamsFile(
            source_path,
            mask_boundary_sample_num=90,
            sample_polar_num=1000,
            sample_point_scale=0.8,
            idx_dtype=torch.int64,
            dtype=torch.float64,
            device='cuda',
        )

        if not mash.toMeshFile(target_path, overwrite=False):
            print("[ERROR][Convertor::convertData]")
            print("\t toMeshFile failed!")
            print("\t target_path:", target_path)
            return False

        return True
