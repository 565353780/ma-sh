import torch
import numpy as np

def toTriangleRotateMatrixs(points: torch.Tensor, triangles: np.ndarray) -> torch.Tensor:
    A = points[triangles[:, 0]]
    B = points[triangles[:, 1]]
    C = points[triangles[:, 2]]

    AB = B - A
    AB = AB / torch.norm(AB, dim=1).reshape(-1, 1)

    AC = C - A
    #AC = AC / torch.norm(AC, dim=1).reshape(-1, 1)

    N = torch.linalg.cross(AB, AC)
    N = N / torch.norm(N, dim=1).reshape(-1, 1)

    regular_AC = torch.linalg.cross(AB, N)
    regular_AC = regular_AC / torch.norm(regular_AC, dim=1).reshape(-1, 1)

    triangle_rotate_matrixs = torch.stack([AB, regular_AC, N], dim=2)

    return triangle_rotate_matrixs
