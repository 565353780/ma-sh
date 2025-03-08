import numpy as np
import open3d as o3d

from ma_sh.Data.mesh import Mesh
from ma_sh.Method.pcd import getPointCloud
from ma_sh.Module.Convertor.base_convertor import BaseConvertor


class Convertor(BaseConvertor):
    def __init__(
        self,
        source_root_folder_path: str,
        target_root_folder_path: str,
        gt_points_num: int = 400000,
        random_weight: float = -1.0,
    ) -> None:
        super().__init__(source_root_folder_path, target_root_folder_path)

        self.gt_points_num = gt_points_num
        self.random_weight = random_weight
        return

    def convertData(self, source_path: str, target_path: str) -> bool:
        mesh = Mesh(source_path)

        if not mesh.isValid():
            print("[ERROR][Convertor::convertData]")
            print("\t mesh is not valid!")
            print("\t source_path:", source_path)
            return False

        if self.random_weight >= 0.0:
            points = mesh.toRandomSamplePoints(self.gt_points_num, self.random_weight)
        #elif mesh.pointNum() > 10 * self.gt_points_num:
        #    points = mesh.points()
        else:
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

        if target_path[-4:] == '.npy':
            np.save(target_path, points)
        else:
            pcd = getPointCloud(points)
            o3d.io.write_point_cloud(target_path, pcd, write_ascii=True)

        return True
