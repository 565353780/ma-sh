import torch
import numpy as np
import open3d as o3d
from typing import Union


def getPointCloud(
    pts: Union[np.ndarray, torch.Tensor],
    normals: Union[np.ndarray, None] = None,
    colors: Union[np.ndarray, None] = None,
):
    if isinstance(pts, torch.Tensor):
        safe_pts = pts.detach().clone().cpu().numpy()
    else:
        safe_pts = pts

    safe_pts = safe_pts.reshape(-1, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(safe_pts.astype(np.float64))
    if normals is not None:
        if normals.shape[0] == pts.shape[0]:
            pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    if colors is not None:
        if colors.shape[0] == pts.shape[0]:
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd
