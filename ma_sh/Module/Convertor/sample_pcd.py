import numpy as np

from ma_sh.Data.mesh import Mesh
from ma_sh.Module.Convertor.base_convertor import BaseConvertor


class Convertor(BaseConvertor):
    def __init__(
        self,
        source_root_folder_path: str,
        target_root_folder_path: str,
        gt_points_num: int = 400000,
    ) -> None:
        super().__init__(source_root_folder_path, target_root_folder_path)

        self.gt_points_num = gt_points_num
        return

    def convertData(self, source_path: str, target_path: str) -> bool:
        mesh = Mesh(source_path)

        if not mesh.isValid():
            print("[ERROR][Convertor::convertData]")
            print("\t mesh is not valid!")
            print("\t source_path:", source_path)
            return False

        try:
            points = mesh.toSamplePoints(self.gt_points_num)
        except KeyboardInterrupt:
            print('[INFO][Convertor::convertData]')
            print('\t program interrupted by the user (Ctrl+C).')
            exit()
        except:
            print("[ERROR][Convertor::convertData]")
            print("\t toSamplePoints failed!")
            print("\t source_path:", source_path)
            return False

        if points is None:
            print("[ERROR][Convertor::convertData]")
            print("\t toSamplePoints failed!")
            print("\t source_path:", source_path)
            return False

        np.save(target_path, points)

        return True
