import torch

from ma_sh.Model.mash import Mash


def demo():
    anchor_num = 40
    mask_degree_max = 5
    sh_degree_max = 3
    mask_boundary_sample_num = 100
    sample_polar_num = 100000
    idx_dtype = torch.int64
    dtype = torch.float64
    device = "cpu"

    mash = Mash(
        anchor_num,
        mask_degree_max,
        sh_degree_max,
        mask_boundary_sample_num,
        sample_polar_num,
        idx_dtype,
        dtype,
        device,
    )

    for i in range(anchor_num):
        mash.mask_params.data[i, 0] = i + 1.0

    for i in range(anchor_num):
        mash.sh_params.data[i, 0] = i + 1.0

    for i in range(anchor_num):
        mash.rotate_vectors.data[i, 0] = i

    for i in range(anchor_num):
        mash.positions.data[i, 0] = i

    sh_points = mash.toSamplePoints()

    print(sh_points.shape)

    mash.renderSamplePoints()
    return True
