import torch
import mash_cpp

from ma_sh.Method.rotate import (
    toRegularRotateVectors,
    toOrthoPosesFromRotateVectors,
    toRotateVectorsFromOrthoPoses
)

def testFaceForwardRotation():
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

def testOrtho6D():
    random_rotate_vectors = torch.randn((10000, 3), dtype=torch.float64, device='cuda')

    regular_rotate_vectors = toRegularRotateVectors(random_rotate_vectors)

    ortho_poses = toOrthoPosesFromRotateVectors(regular_rotate_vectors)

    new_rotate_vectors = toRotateVectorsFromOrthoPoses(ortho_poses)

    error = torch.norm(new_rotate_vectors - regular_rotate_vectors, dim=1)

    error_idxs = torch.where(error > 1e-6)

    if error_idxs[0].shape[0] > 0:
        err = error[error_idxs]

        error_rv0 = random_rotate_vectors[error_idxs]
        error_rv1 = regular_rotate_vectors[error_idxs]
        error_rv2 = new_rotate_vectors[error_idxs]

        rot_mtx0 = mash_cpp.toRotateMatrixs(error_rv0)
        rot_mtx1 = mash_cpp.toRotateMatrixs(error_rv1)
        rot_mtx2 = mash_cpp.toRotateMatrixs(error_rv2)

        print('error:')
        print(err)

        print('source_rotate_vectors:')
        print(error_rv0)
        print('regular_rotate_vectors:')
        print(error_rv1)
        print('new_rotate_vectors:')
        print(error_rv2)

        print('source_rotate_matrixs:')
        print(rot_mtx0)
        print('rotate_matrixs:')
        print(rot_mtx1)
        print('new_rotate_matrixs:')
        print(rot_mtx2)
        exit()

    max_error = torch.max(error)

    assert max_error < 1e-6
    return True

def test():
    testFaceForwardRotation()

    testOrtho6D()

    return True
