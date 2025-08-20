import numpy as np
import open3d as o3d
from math import ceil
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


def downSample(
    pcd: o3d.geometry.PointCloud, sample_point_num, try_fps_first: bool = True
):
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
