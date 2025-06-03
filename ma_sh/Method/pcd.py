import numpy as np
import open3d as o3d
from math import ceil
from typing import Union

from ma_sh.Data.abb import ABB


def getPointCloud(
    pts: np.ndarray,
    normals: Union[np.ndarray, None] = None,
    colors: Union[np.ndarray, None] = None,
):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    if normals is not None:
        if normals.shape[0] == pts.shape[0]:
            pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    if colors is not None:
        if colors.shape[0] == pts.shape[0]:
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd


def downSample(pcd, sample_point_num, try_fps_first: bool = True):
    if sample_point_num < 1:
        print("[WARN][pcd::downSample]")
        print("\t sample_point_num < 1!")
        return None

    if try_fps_first:
        try:
            down_sample_pcd = pcd.farthest_point_down_sample(sample_point_num)
        except KeyboardInterrupt:
            print("[INFO][pcd::downSample]")
            print("\t program interrupted by the user (Ctrl+C).")
            exit()
        except:
            every_k_points = ceil(np.asarray(pcd.points).shape[0] / sample_point_num)
            down_sample_pcd = pcd.uniform_down_sample(every_k_points)
    else:
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

    normals_1 = np.asarray(pcd_1.normals)
    normals_2 = np.asarray(pcd_2.normals)

    colors_1 = np.asarray(pcd_1.colors)
    colors_2 = np.asarray(pcd_2.colors)

    merged_points = np.vstack([points_1, points_2])
    merged_normals = np.vstack([normals_1, normals_2])
    merged_colors = np.vstack([colors_1, colors_2])

    merged_pcd = getPointCloud(merged_points, merged_normals, merged_colors)

    return merged_pcd
