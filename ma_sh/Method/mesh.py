import numpy as np
import open3d as o3d
from typing import Union, Tuple


def samplePointCloud(
    mesh: o3d.geometry.TriangleMesh,
    sample_point_num: int,
) -> o3d.geometry.PointCloud:
    if sample_point_num < 1:
        print("[ERROR][mesh::samplePointCloud]")
        print("\t sample_point_num < 1!")
        return None

    return mesh.sample_points_poisson_disk(sample_point_num, use_triangle_normal=True)


def samplePoints(
    mesh: o3d.geometry.TriangleMesh,
    sample_point_num: int,
    with_color=False,
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray, None]:
    pcd = samplePointCloud(mesh, sample_point_num)

    if pcd is None:
        print("[ERROR][mesh::samplePoints]")
        print("\t samplePointCloud failed!")
        return None

    if with_color:
        return np.array(pcd.points), np.array(pcd.colors)
    return np.array(pcd.points)
