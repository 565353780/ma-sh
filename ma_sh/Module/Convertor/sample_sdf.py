from sdf_generate.Method.sample_sdf import convertSDFNearSurface

from ma_sh.Module.Convertor.base_convertor import BaseConvertor


class Convertor(BaseConvertor):
    def __init__(
        self,
        source_root_folder_path: str,
        target_root_folder_path: str,
        sample_sdf_point_num: int = 250000,
        gauss_noise: float = 0.0025,
    ) -> None:
        super().__init__(source_root_folder_path, target_root_folder_path)

        self.sample_sdf_point_num = sample_sdf_point_num
        self.gauss_noise = gauss_noise
        return

    def convertData(self, source_path: str, target_path: str) -> bool:
        try:
            convertSDFNearSurface(
                source_path,
                target_path,
                self.sample_sdf_point_num,
                self.gauss_noise,
                True,
            )
        except:
            print("[ERROR][Convertor::convertData]")
            print("\t convertSDFNearSurface failed!")
            print("\t source_path:", source_path)
            return False

        return True
