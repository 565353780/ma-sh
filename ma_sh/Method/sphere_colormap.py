import numpy as np
import open3d as o3d
from typing import Union
from matplotlib import cm
from functools import partial

from spherical_harmonics.Data.sh_3d_model import SH3DModel


def gaussian_dist_func(phi, theta):
    return np.exp(-((phi - np.pi/2)**2 + (theta - np.pi)**2) / 0.2)

def createColoredSphere(
    distance_func,
    resolution: int = 200,
    dist_min: Union[float, None] = None,
    dist_max: Union[float, None] = None,
) -> o3d.geometry.TriangleMesh:
    sphere = o3d.geometry.TriangleMesh.create_sphere(
        radius=1.0,
        resolution=resolution,
    )
    sphere.compute_vertex_normals()

    verts = np.asarray(sphere.vertices)
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    r = np.linalg.norm(verts, axis=1)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    phi[phi < 0] += 2 * np.pi

    values = distance_func(phi, theta)

    if dist_min is None or dist_max is None:
        dist_min = values.min()
        dist_max = values.max()

    values_norm = (values - dist_min) / (dist_max - dist_min)

    cmap = cm.get_cmap("viridis")
    colors = cmap(values_norm)[:, :3]

    sphere.vertex_colors = o3d.utility.Vector3dVector(colors)

    return sphere

def createSHColoredSphere(
    degree_max: int,
    sh_params: Union[np.ndarray, list],
    resolution: int = 200,
    dist_min: Union[float, None] = None,
    dist_max: Union[float, None] = None,
) -> o3d.geometry.TriangleMesh:
    sh_3d_model = SH3DModel(degree_max, 'numpy')
    sh_3d_model.setParams(sh_params)

    dist_func = partial(sh_3d_model.getDiffValues, method_name='numpy')

    sh_colored_sphere = createColoredSphere(dist_func, resolution, dist_min, dist_max)

    return sh_colored_sphere

def createGTColoredSphere(
    mesh_file_path: str,
    position: Union[np.ndarray, list],
    resolution: int = 200,
    dist_min: Union[float, None] = None,
    dist_max: Union[float, None] = None,
) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(mesh_file_path)
    mesh.compute_vertex_normals()

    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)

    p = np.asarray(position, dtype=np.float32)

    sphere = o3d.geometry.TriangleMesh.create_sphere(
        radius=1.0,
        resolution=resolution,
    )
    sphere.compute_vertex_normals()

    verts = np.asarray(sphere.vertices)
    origins = np.tile(p, (verts.shape[0], 1)).astype(np.float32)

    rays = o3d.core.Tensor(np.hstack((origins, verts)), dtype=o3d.core.Dtype.Float32)

    ans = scene.cast_rays(rays)
    depth_map = ans['t_hit'].numpy()

    depth_map[np.isinf(depth_map)] = np.nan
    depth_map[np.isnan(depth_map)] = 0.0

    if dist_min is None or dist_max is None:
        dist_min = depth_map.min()
        dist_max = depth_map.max()

    depth_norm = (depth_map - dist_min) / (dist_max - dist_min)

    cmap = cm.get_cmap("viridis")
    colors = cmap(depth_norm)[:, :3]

    sphere.vertex_colors = o3d.utility.Vector3dVector(colors)

    return sphere
