import torch
from tqdm import trange

from ma_sh.Method.mash_unit import (
    toParams,
    toPreLoadUniformSamplePolars,
    toPreLoadMaskBoundaryIdxs,
    toPreLoadBaseValues,
    toPreLoadSHDirections,
    toMaskBoundaryThetas,
    toInMaxMaskIdxs,
    toInMaxMaskPolars,
    toInMaxMaskThetas,
    toInMaskSamplePolarWeights,
    toSamplePolars,
    toSHValues,
    toSHPoints,
)


def test():
    anchor_num = 4
    mask_degree_max = 5
    sh_degree_max = 3
    mask_boundary_sample_num = 10
    sample_polar_num = 10
    idx_dtype = torch.int64
    dtype = torch.float64
    device = "cpu"

    mask_params, sh_params, rotate_vectors, positions = toParams(
        anchor_num, mask_degree_max, sh_degree_max, dtype, device
    )
    sample_phis, sample_thetas = toPreLoadUniformSamplePolars(
        sample_polar_num, dtype, device
    )
    mask_boundary_phi_idxs = toPreLoadMaskBoundaryIdxs(
        anchor_num, mask_boundary_sample_num, idx_dtype, device
    )
    mask_boundary_base_values, sample_base_values = toPreLoadBaseValues(
        anchor_num,
        mask_boundary_sample_num,
        mask_degree_max,
        sample_phis,
    )
    sample_sh_directions = toPreLoadSHDirections(sample_phis, sample_thetas)

    for _ in trange(1000):
        mask_boundary_thetas = toMaskBoundaryThetas(
            mask_params, mask_boundary_base_values, mask_boundary_phi_idxs
        )
        (
            in_max_mask_sample_polar_idxs,
            in_max_mask_sample_polar_data_idxs,
        ) = toInMaxMaskIdxs(sample_thetas, mask_boundary_thetas, mask_boundary_phi_idxs)
        (
            in_max_mask_sample_phis,
            in_max_mask_sample_thetas,
        ) = toInMaxMaskPolars(
            sample_phis, sample_thetas, in_max_mask_sample_polar_data_idxs
        )
        (
            in_max_mask_base_values,
            in_max_mask_thetas,
        ) = toInMaxMaskThetas(
            mask_params,
            sample_base_values,
            in_max_mask_sample_polar_idxs,
            in_max_mask_sample_polar_data_idxs,
        )
        (
            in_mask_sample_phis,
            in_mask_sample_polar_idxs,
            in_mask_sample_polar_data_idxs,
            in_mask_base_values,
            in_mask_sample_theta_weights,
        ) = toInMaskSamplePolarWeights(
            mask_params,
            in_max_mask_base_values,
            in_max_mask_thetas,
            in_max_mask_sample_phis,
            in_max_mask_sample_thetas,
            in_max_mask_sample_polar_idxs,
            in_max_mask_sample_polar_data_idxs,
        )
        detect_thetas = toSamplePolars(
            mask_params,
            in_mask_base_values,
            in_mask_sample_polar_idxs,
            in_mask_sample_theta_weights,
        )
        sh_values = toSHValues(
            sh_degree_max,
            sh_params,
            in_mask_sample_phis,
            detect_thetas,
            in_mask_sample_polar_idxs,
        )
        sh_points = toSHPoints(
            rotate_vectors,
            positions,
            sample_sh_directions,
            sh_values,
            in_mask_sample_polar_idxs,
            in_mask_sample_polar_data_idxs,
        )

    return True
