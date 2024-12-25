import torch
import numpy as np

import mash_cpp


def toTriangleRotateMatrixs(points: torch.Tensor, triangles: np.ndarray) -> torch.Tensor:
    A = points[triangles[:, 0]]
    B = points[triangles[:, 1]]
    C = points[triangles[:, 2]]

    AB = B - A
    AB = AB / torch.norm(AB, dim=1).reshape(-1, 1)

    AC = C - A

    N = torch.linalg.cross(AB, AC)
    N = N / torch.norm(N, dim=1).reshape(-1, 1)

    regular_AC = torch.linalg.cross(AB, N)
    regular_AC = regular_AC / torch.norm(regular_AC, dim=1).reshape(-1, 1)

    triangle_rotate_matrixs = torch.stack([AB, regular_AC, N], dim=2)

    return triangle_rotate_matrixs

def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))
    v_mag = torch.max(
        v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).to(v.device)))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    return v

def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)),
                    1)

    return out

def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[:, 0:3]
    y_raw = poses[:, 3:6]

    x = normalize_vector(x_raw)
    z = cross_product(x, y_raw)
    z = normalize_vector(z)
    y = cross_product(z, x)

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)
    return matrix

def toRegularRotateVectors(rotate_vectors: torch.Tensor) -> torch.Tensor:
    source_dtype = rotate_vectors.dtype

    valid_rotate_vectors = rotate_vectors.type(torch.float64)

    rotate_matrixs = mash_cpp.toRotateMatrixs(valid_rotate_vectors)

    regular_rotate_vectors = mash_cpp.toRotateVectors(rotate_matrixs)

    valid_regular_rotate_vectors = regular_rotate_vectors.type(source_dtype)

    return valid_regular_rotate_vectors

def toOrthoPosesFromRotateVectors(rotate_vectors: torch.Tensor) -> torch.Tensor:
    source_dtype = rotate_vectors.dtype

    valid_rotate_vectors = rotate_vectors.type(torch.float64)

    rotate_matrixs = mash_cpp.toRotateMatrixs(valid_rotate_vectors)

    ortho_poses = rotate_matrixs.permute(0, 2, 1).reshape(-1, 9)[:, :6]

    valid_ortho_poses = ortho_poses.type(source_dtype)

    return valid_ortho_poses

def toRotateVectorsFromOrthoPoses(ortho_poses: torch.Tensor) -> torch.Tensor:
    source_dtype = ortho_poses.dtype

    valid_ortho_poses = ortho_poses.type(torch.float64)

    rotate_matrixs = compute_rotation_matrix_from_ortho6d(valid_ortho_poses)

    rotate_vectors = mash_cpp.toRotateVectors(rotate_matrixs)

    valid_rotate_vectors = rotate_vectors.type(source_dtype)

    return valid_rotate_vectors
