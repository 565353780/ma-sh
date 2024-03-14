import numpy as np


def samplePointCloud(mesh, sample_point_num):
    if sample_point_num < 1:
        print("[ERROR][mesh::samplePointCloud]")
        print("\t sample_point_num < 1!")
        return None

    return mesh.sample_points_poisson_disk(sample_point_num, use_triangle_normal=True)


def samplePoints(mesh, sample_point_num, with_color=False):
    pcd = samplePointCloud(mesh, sample_point_num)

    if pcd is None:
        print("[ERROR][mesh::samplePoints]")
        print("\t samplePointCloud failed!")
        return None

    if with_color:
        return np.array(pcd.points), np.array(pcd.colors)
    return np.array(pcd.points)
