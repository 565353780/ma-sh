from typing import Union
from copy import deepcopy
import numpy as np
import open3d as o3d

from ma_sh.Data.abb import ABB

from ma_sh.Method.io import loadPcdFile
from ma_sh.Method.pcd import getPointCloud, downSample, getCropPointCloud
from ma_sh.Method.render import renderGeometries

class Pcd(object):
    def __init__(self, pcd_file_path=None) -> None:
        self.pcd = None

        if pcd_file_path is not None:
            self.loadPcd(pcd_file_path)
        return

    @classmethod
    def from_numpy(cls, points: np.ndarray):
        pcd = Pcd()
        pcd.pcd = o3d.geometry.PointCloud()
        pcd.pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    @classmethod
    def from_o3d(cls, point_cloud: o3d.geometry.PointCloud):
        pcd = Pcd()
        pcd.pcd = point_cloud
        return pcd

    def reset(self) -> bool:
        self.pcd = None
        return True

    def loadPcd(self, pcd_file_path: str) -> bool:
        self.reset()

        self.pcd = loadPcdFile(pcd_file_path)

        if self.pcd is None:
            print("[ERROR][Pcd::loadPcd]")
            print('\t loadPcdFile failed!')
            return False

        return True

    def isValid(self, output_info: bool=False) -> bool:
        if self.pcd is None:
            if output_info:
                print("[ERROR][Pcd::isValid]")
                print('\t pcd is None! please load pcd first!')
            return False

        return True

    def getPoints(self) -> Union[None, np.ndarray]:
        if self.pcd is None:
            print("[ERROR][Pcd::getPoints]")
            print('\t pcd is None! please load pcd first!')
            return None

        return np.array(self.pcd.points)

    def toResolution(self, sample_point_num: int):
        if not self.isValid(True):
            print("[ERROR][Pcd::toResolution]")
            print('\t isValid failed!')
            return None

        new_pcd = downSample(self.pcd, sample_point_num)

        if new_pcd is None:
            print("[ERROR][Pcd::toResolution]")
            print('\t downSample failed!')
            return None

        return Pcd.from_o3d(new_pcd)

    def cropByABB(self, abb: Union[None, ABB]):
        if abb is None:
            return Pcd.from_o3d(deepcopy(self.pcd))

        return Pcd.from_o3d(getCropPointCloud(self.pcd, abb))

    def cropByDist(self, center: np.ndarray, dist: float):
        if dist <= 0:
            print('[ERROR][Pcd::cropByDist]')
            print('\t dist <= 0!')
            return None

        abb = ABB.from_numpy([center - dist, center + dist])

        crop_pcd = self.cropByABB(abb)

        points = crop_pcd.getPoints()

        if points is None:
            print('[ERROR][Pcd::cropByDist]')
            print('\t getPoints failed!')
            return None

        dists = np.linalg.norm(points - center, ord=2, axis=1)

        crop_idxs = np.where(dists <= dist)[0]

        crop_points = points[crop_idxs]

        return Pcd.from_numpy(crop_points)

    def getDist(self, pts: Union[list, np.ndarray]):
        return_value = False
        if isinstance(pts, list) and len(pts) == 3:
            pts = np.array([pts])
            return_value = True

        pts = np.array(pts)

        if len(pts.shape) == 1:
            pts = np.array([pts])
            return_value = True

        query_pcd = getPointCloud(pts)
        dists = np.array(query_pcd.compute_point_cloud_distance(self.pcd))

        if return_value:
            return dists[0]

        return dists

    def render(self):
        if not self.isValid(True):
            print("[ERROR][Pcd::render]")
            print('\t isValid failed!')
            return False

        renderGeometries(self.pcd, "Pcd")
        return True
