import torch
import mash_cpp


def test():
    a = torch.randn(1000, 3).type(torch.float64)
    a_norm = torch.norm(a, p=2, dim=1).reshape(-1, 1)
    a = a / a_norm

    z_axis = torch.zeros_like(a)
    z_axis[:, 2] = 1.0

    b = mash_cpp.toRotateVectorsByFaceForwardVectors(a)

    rotate_matrixs = mash_cpp.toRotateMatrixs(b)

    v_z_axis = z_axis.reshape(-1, 3, 1)
    trans_to_a = torch.matmul(rotate_matrixs, v_z_axis).reshape(-1, 3)

    error = torch.abs(a - trans_to_a)
    max_error = torch.max(error)

    assert max_error < 1e-6
    return True
