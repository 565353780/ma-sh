import torch
import numpy as np
from typing import Tuple

def toOuterCircles(points: torch.Tensor, triangles: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    A = points[triangles[:, 0]]
    B = points[triangles[:, 1]]
    C = points[triangles[:, 2]]

    x1 = A[:, 0]
    y1 = A[:, 1]
    z1 = A[:, 2]

    x2 = B[:, 0]
    y2 = B[:, 1]
    z2 = B[:, 2]

    x3 = C[:, 0]
    y3 = C[:, 1]
    z3 = C[:, 2]

    a1 = y1*z2 - y2*z1 - y1*z3 + y3*z1 + y2*z3 - y3*z2
    b1 = -(x1*z2 - x2*z1 - x1*z3 + x3*z1 + x2*z3 - x3*z2)
    c1 = x1*y2 - x2*y1 - x1*y3 + x3*y1 + x2*y3 - x3*y2
    d1 = -(x1*y2*z3 - x1*y3*z2 - x2*y1*z3 + x2*y3*z1 + x3*y1*z2 - x3*y2*z1)

    a2 = 2 * (x2 - x1)
    b2 = 2 * (y2 - y1)
    c2 = 2 * (z2 - z1)
    d2 = x1 * x1 + y1 * y1 + z1 * z1 - x2 * x2 - y2 * y2 - z2 * z2

    a3 = 2 * (x3 - x1)
    b3 = 2 * (y3 - y1)
    c3 = 2 * (z3 - z1)
    d3 = x1 * x1 + y1 * y1 + z1 * z1 - x3 * x3 - y3 * y3 - z3 * z3

    center_x = -(b1*c2*d3 - b1*c3*d2 - b2*c1*d3 + b2*c3*d1 + b3*c1*d2 - b3*c2*d1) / (a1*b2*c3 - a1*b3*c2 - a2*b1*c3 + a2*b3*c1 + a3*b1*c2 - a3*b2*c1)
    center_y = (a1*c2*d3 - a1*c3*d2 - a2*c1*d3 + a2*c3*d1 + a3*c1*d2 - a3*c2*d1) / (a1*b2*c3 - a1*b3*c2 - a2*b1*c3 + a2*b3*c1 + a3*b1*c2 - a3*b2*c1)
    center_z = -(a1*b2*d3 - a1*b3*d2 - a2*b1*d3 + a2*b3*d1 + a3*b1*d2 - a3*b2*d1) / (a1*b2*c3 - a1*b3*c2 - a2*b1*c3 + a2*b3*c1 + a3*b1*c2 - a3*b2*c1)

    centers = torch.vstack([center_x, center_y, center_z]).permute(1, 0)

    radius = torch.norm(A - centers, dim=1)

    return centers, radius

def toOuterEllipses(points: torch.Tensor, triangles: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if triangles.shape[0] == 0:
        print('[ERROR][outer::toOuterEllipses]')
        print('\t triangles is empty!')
        return torch.empty(0), torch.empty(0), torch.empty(0)

    # 获取三角形的顶点坐标
    triangle_points = points[triangles]  # 形状为 (num_triangles, 3, 3)

    # 计算每个三角形的中心
    triangle_centers = triangle_points.mean(dim=1)  # 形状为 (num_triangles, 3)

    # 中心化顶点坐标
    centered_points = triangle_points - triangle_centers.unsqueeze(1)  # 形状为 (num_triangles, 3, 3)

    # 计算协方差矩阵
    cov_matrices = torch.einsum('bij,bik->bjk', centered_points, centered_points) / 2  # 形状为 (num_triangles, 3, 3)

    # 对协方差矩阵进行特征值分解
    eigvals, eigvecs = torch.linalg.eigh(cov_matrices)  # eigvals形状为 (num_triangles, 3)，eigvecs形状为 (num_triangles, 3, 3)

    # 获取长轴和短轴长度
    #FIXME: here ellipses are not too accurate, need to upgrade this algo
    axes_lengths = 1.2 * torch.sqrt(eigvals[:, [2, 1]])  # 形状为 (num_triangles, 2)，只取最大的两个特征值对应的轴

    # 获取旋转矩阵（取对应于长轴和短轴的特征向量）
    rotation_matrices = eigvecs[:, :, [2, 1]]  # 形状为 (num_triangles, 3, 2)
    rotation_matrices = torch.cat((rotation_matrices, torch.linalg.cross(rotation_matrices[:, :, 0], rotation_matrices[:, :, 1]).unsqueeze(2)), dim=2)  # 形状为 (num_triangles, 3, 3)

    return triangle_centers, axes_lengths, rotation_matrices
