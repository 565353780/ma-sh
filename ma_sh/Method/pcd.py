import numpy as np
import open3d as o3d
from math import ceil
from typing import Union

from ma_sh.Data.abb import ABB


def getPointCloud(pts: np.ndarray, normals: Union[np.ndarray, None]=None, colors: Union[np.ndarray, None]=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if normals is not None:
        if normals.shape[0] == pts.shape[0]:
            pcd.normals = o3d.utility.Vector3dVector(normals)
    if colors is not None:
        if colors.shape[0] == pts.shape[0]:
            pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def downSample(pcd, sample_point_num):
    if sample_point_num < 1:
        print("[WARN][pcd::downSample]")
        print("\t sample_point_num < 1!")
        return None

    try:
        down_sample_pcd = pcd.farthest_point_down_sample(sample_point_num)
    except KeyboardInterrupt:
        print('[INFO][pcd::downSample]')
        print('\t program interrupted by the user (Ctrl+C).')
        exit()
    except:
        every_k_points = ceil(np.asarray(pcd.points).shape[0] / sample_point_num)
        down_sample_pcd = pcd.uniform_down_sample(every_k_points)

    return down_sample_pcd


def getCropPointCloud(
    pcd: o3d.geometry.PointCloud, abb: ABB
) -> o3d.geometry.PointCloud:
    o3d_abb = abb.toOpen3DABB()
    return pcd.crop(o3d_abb)

def toMergedPcd(
    pcd_1: o3d.geometry.PointCloud,
    pcd_2: o3d.geometry.PointCloud,
) -> o3d.geometry.PointCloud:
    points_1 = np.asarray(pcd_1.points)
    points_2 = np.asarray(pcd_2.points)

    merged_points = np.vstack([points_1, points_2])
    merged_pcd = getPointCloud(merged_points)

    return merged_pcd
