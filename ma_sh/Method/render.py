import numpy as np
import open3d as o3d
from typing import Union


def translateGeometries(
    translate: Union[list, np.ndarray], geometry_list: list
) -> bool:
    translate = np.ndarray(translate, dtype=float)

    for geometry in geometry_list:
        geometry.translate(translate)
    return True


def getLineSet(
    start: Union[list, np.ndarray],
    vectors: Union[list, np.ndarray],
    color: Union[list, np.ndarray],
) -> o3d.geometry.PointCloud:
    start = np.array(start, dtype=float)
    vectors = np.array(vectors, dtype=float)
    color = np.array(color, dtype=float)

    points = np.vstack([start.reshape(1, -1), start + vectors])
    lines = np.zeros([vectors.shape[0], 2], dtype=int)
    lines[:, 1] = np.arange(1, points.shape[0])
    colors = np.zeros([vectors.shape[0], 3], dtype=float)
    colors[:, :] = color

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def getMaskConeMesh(
    anchor_position: Union[list, np.ndarray],
    mask_boundary_points: Union[list, np.ndarray],
    color: Union[list, np.ndarray],
) -> o3d.geometry.TriangleMesh:
    anchor_position = np.array(anchor_position, dtype=float)
    mask_boundary_points = np.array(mask_boundary_points, dtype=float)
    color = np.array(color, dtype=float)

    points = np.vstack([anchor_position.reshape(1, 3), mask_boundary_points])
    triangles = np.zeros([2 * mask_boundary_points.shape[0], 3], dtype=int)
    triangles[: mask_boundary_points.shape[0], 1] = np.arange(1, points.shape[0])
    triangles[: mask_boundary_points.shape[0], 2] = np.arange(2, points.shape[0] + 1)
    triangles[mask_boundary_points.shape[0] - 1, 2] = 1
    triangles[mask_boundary_points.shape[0] :, 1] = np.arange(2, points.shape[0] + 1)
    triangles[mask_boundary_points.shape[0] :, 2] = np.arange(1, points.shape[0])
    triangles[2 * mask_boundary_points.shape[0] - 1, 1] = 1

    mask_cone_mesh = o3d.geometry.TriangleMesh()
    mask_cone_mesh.vertices = o3d.utility.Vector3dVector(points)
    mask_cone_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mask_cone_mesh.paint_uniform_color(color)

    mask_cone_mesh.compute_vertex_normals()
    mask_cone_mesh.compute_triangle_normals()
    return mask_cone_mesh


def renderGeometries(geometry_list, window_name="Geometry List"):
    if not isinstance(geometry_list, list):
        geometry_list = [geometry_list]

    o3d.visualization.draw_geometries(geometry_list, window_name)
    return True


def renderPoints(points: np.ndarray, window_name="Points"):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return renderGeometries(pcd, window_name)
