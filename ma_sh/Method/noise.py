import os
import numpy as np
import open3d as o3d

def generate_sparse_incomplete_noisy_pcd(
    mesh_file_path: str,
    num_points: int=20000,
    missing_ratio: float=0.5,
    noise_std: float=0.01,
) -> o3d.geometry.PointCloud:
    if not os.path.exists(mesh_file_path):
        print('[ERROR][noise::generate_sparse_incomplete_noisy_pcd]')
        print('\t mesh file not exist!')
        print('\t mesh_file_path:', mesh_file_path)
        return o3d.geometry.PointCloud()

    mesh = o3d.io.read_triangle_mesh(mesh_file_path)
    mesh.compute_vertex_normals()

    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    points = np.asarray(pcd.points)

    num_missing = int(num_points * missing_ratio)
    missing_idx = np.random.choice(num_points, num_missing, replace=False)
    mask = np.ones(num_points, dtype=bool)
    mask[missing_idx] = False

    sparse_points = points[mask]

    noise = np.random.normal(scale=noise_std, size=sparse_points.shape)
    noisy_points = sparse_points + noise

    sparse_pcd = o3d.geometry.PointCloud()
    sparse_pcd.points = o3d.utility.Vector3dVector(noisy_points)

    return sparse_pcd
