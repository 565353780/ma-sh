import torch
from tqdm import trange

from ma_sh.Method.Mash.mash import toParams, toPreLoadDatas, toMashSamplePoints


def test():
    anchor_num = 4
    mask_degree_max = 5
    sh_degree_max = 3
    mask_boundary_sample_num = 10
    sample_polar_num = 10
    idx_dtype = torch.int64
    dtype = torch.float64
    device = "cpu"
    (
        mask_params,
        sh_params,
        rotate_vectors,
        positions,
    ) = toParams(
        anchor_num,
        mask_degree_max,
        sh_degree_max,
        dtype,
        device,
    )

    for i in range(anchor_num):
        mask_params.data[i, 0] = i + 1.0

    for i in range(anchor_num):
        sh_params.data[i, 0] = i + 1.0

    for i in range(anchor_num):
        rotate_vectors.data[i, 0] = i

    for i in range(anchor_num):
        positions.data[i, 0] = i

    (
        sample_phis,
        sample_thetas,
        mask_boundary_phis,
        mask_boundary_phi_idxs,
        mask_boundary_phi_data_idxs,
        mask_boundary_base_values,
        sample_base_values,
        sample_sh_directions,
    ) = toPreLoadDatas(
        anchor_num,
        mask_degree_max,
        mask_boundary_sample_num,
        sample_polar_num,
        idx_dtype,
        dtype,
        device,
    )

    for _ in trange(1000):
        sh_points = toMashSamplePoints(
            sh_degree_max,
            mask_params,
            sh_params,
            rotate_vectors,
            positions,
            sample_phis,
            sample_thetas,
            mask_boundary_phis,
            mask_boundary_phi_idxs,
            mask_boundary_phi_data_idxs,
            mask_boundary_base_values,
            sample_base_values,
            sample_sh_directions,
        )

    return True
