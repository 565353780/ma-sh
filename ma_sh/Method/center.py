import torch
import numpy as np
from typing import Tuple

def toOuterCenters(points: torch.Tensor, triangles: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
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
