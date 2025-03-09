import numpy as np
import open3d as o3d
from typing import Union

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
        need_normalize: bool = False,
    ) -> None:
        super().__init__(source_root_folder_path, target_root_folder_path)

        self.gt_points_num = gt_points_num
        self.random_weight = random_weight
        self.need_normalize = need_normalize
        return

    def convertMeshToPoints(self, source_path: str) -> Union[np.ndarray, None]:
        mesh = Mesh(source_path)

        if self.need_normalize:
            mesh.normalize()

        if not mesh.isValid():
            print("[ERROR][Convertor::convertMeshToPoints]")
            print("\t mesh is not valid!")
            print("\t source_path:", source_path)
            return None

        if self.random_weight >= 0.0:
            points = mesh.toRandomSamplePoints(self.gt_points_num, self.random_weight)
            return points

        #if mesh.pointNum() > 10 * self.gt_points_num:
        #    points = mesh.points()
        #    return points

        try:
            points = mesh.toSamplePoints(self.gt_points_num)
            return points
        except KeyboardInterrupt:
            print('[INFO][Convertor::convertMeshToPoints]')
            print('\t program interrupted by the user (Ctrl+C).')
            exit()
        except:
            print("[ERROR][Convertor::convertMeshToPoints]")
            print("\t toSamplePoints failed!")
            print("\t source_path:", source_path)
            return None

    def convertBinToPoints(self, source_path: str) -> Union[np.ndarray, None]:
        data = np.fromfile(source_path, dtype=np.float32).reshape(-1, 4)
        points = data[:, :3]

        if self.need_normalize:
            min_bound = np.min(points, axis=0)
            max_bound = np.max(points, axis=0)
            length = np.max(max_bound - min_bound)
            scale = 0.9 / length
            center = (min_bound + max_bound) / 2.0

            points = (points - center) * scale

        return points

    def convertData(self, source_path: str, target_path: str) -> bool:
        source_type = source_path.split('.')[-1]
        if source_type in ['obj', 'ply']:
            points = self.convertMeshToPoints(source_path)
        elif source_type in ['bin']:
            points = self.convertBinToPoints(source_path)

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
