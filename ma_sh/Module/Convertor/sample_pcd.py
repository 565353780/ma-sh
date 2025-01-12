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
        '''
        if '000-091' not in source_path:
            return True

        objaverse_shape_id_list = [
            '000-091/bf193e241b2f48f0bd2208f89e38fae8',
            '000-091/91979ad79916460d92c7697464f2b5f4',
            '000-091/d4efa3e396274421b07b2fa4314c60bb',
            '000-091/97c493d5c7a443b89229e5f7edb3ae4a',
            '000-091/01fcb4e4c36548ca86624b63dfc6b255',
            '000-091/9df219962230449caa4c95a60feb0c9e',
        ]
        valid_shape = False
        for shape_id in objaverse_shape_id_list:
            if shape_id in source_path:
                valid_shape = True
                break

        if not valid_shape:
            return True
        '''

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
