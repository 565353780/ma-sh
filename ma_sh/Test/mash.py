import torch
from tqdm import trange

from ma_sh.Config.mode import DEBUG

from ma_sh.Method.kernel import toParams, toPreLoadDatas, toMashSamplePoints


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

    if DEBUG:
        mask_params.requires_grad_(False)
    for i in range(anchor_num):
        mask_params[i, 0] = i + 1.0
    mask_params.requires_grad_(True)

    if DEBUG:
        sh_params.requires_grad_(False)
    for i in range(anchor_num):
        sh_params[i, 0] = i + 1.0
    sh_params.requires_grad_(True)

    if DEBUG:
        rotate_vectors.requires_grad_(False)
    for i in range(anchor_num):
        rotate_vectors[i, 0] = i
    rotate_vectors.requires_grad_(True)

    if DEBUG:
        positions.requires_grad_(False)
    for i in range(anchor_num):
        positions[i, 0] = i
    positions.requires_grad_(True)

    (
        sample_phis,
        sample_thetas,
        mask_boundary_phi_idxs,
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
            mask_boundary_phi_idxs,
            mask_boundary_base_values,
            sample_base_values,
            sample_sh_directions,
        )

    return True
