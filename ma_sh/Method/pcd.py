from math import ceil
import numpy as np
import open3d as o3d

from ma_sh.Data.abb import ABB


def getPointCloud(pts: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def downSample(pcd, sample_point_num):
    if sample_point_num < 1:
        print("[WARN][pcd::downSample]")
        print("\t sample_point_num < 1!")
        return None

    try:
        down_sample_pcd = pcd.farthest_point_down_sample(sample_point_num)
    except:
        every_k_points = ceil(np.asarray(pcd.points).shape[0] / sample_point_num)
        down_sample_pcd = pcd.uniform_down_sample(every_k_points)

    return down_sample_pcd


def getCropPointCloud(
    pcd: o3d.geometry.PointCloud, abb: ABB
) -> o3d.geometry.PointCloud:
    o3d_abb = abb.toOpen3DABB()
    return pcd.crop(o3d_abb)
