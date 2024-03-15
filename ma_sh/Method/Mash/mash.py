import torch

from ma_sh.Method.Mash.mash_unit import (
    toInMaxMaskBaseValues,
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


def toPreLoadDatas(
    anchor_num: int,
    mask_degree_max: int,
    mask_boundary_sample_num: int,
    sample_polar_num: int,
    idx_dtype,
    dtype,
    device: str,
):
    sample_phis, sample_thetas = toPreLoadUniformSamplePolars(
        sample_polar_num, dtype, device
    )
    mask_boundary_phi_idxs, mask_boundary_phi_data_idxs = toPreLoadMaskBoundaryIdxs(
        anchor_num, mask_boundary_sample_num, idx_dtype, device
    )
    mask_boundary_phis, mask_boundary_base_values, sample_base_values = (
        toPreLoadBaseValues(
            anchor_num, mask_boundary_sample_num, mask_degree_max, sample_phis
        )
    )
    sample_sh_directions = toPreLoadSHDirections(sample_phis, sample_thetas)

    return (
        sample_phis,
        sample_thetas,
        mask_boundary_phis,
        mask_boundary_phi_idxs,
        mask_boundary_phi_data_idxs,
        mask_boundary_base_values,
        sample_base_values,
        sample_sh_directions,
    )


def toMashSamplePoints(
    sh_degree_max: int,
    mask_params: torch.Tensor,
    sh_params: torch.Tensor,
    rotate_vectors: torch.Tensor,
    positions: torch.Tensor,
    sample_phis: torch.Tensor,
    sample_thetas: torch.Tensor,
    mask_boundary_phis: torch.Tensor,
    mask_boundary_phi_idxs: torch.Tensor,
    mask_boundary_phi_data_idxs: torch.Tensor,
    mask_boundary_base_values: torch.Tensor,
    sample_base_values: torch.Tensor,
    sample_sh_directions: torch.Tensor,
) -> torch.Tensor:
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
    in_max_mask_base_values = toInMaxMaskBaseValues(
        sample_base_values, in_max_mask_sample_polar_data_idxs
    )
    in_max_mask_thetas = toInMaxMaskThetas(
        mask_params,
        in_max_mask_base_values,
        in_max_mask_sample_polar_idxs,
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
    all_sample_phis = torch.hstack([mask_boundary_phis, in_mask_sample_phis])
    all_sample_thetas = torch.hstack([mask_boundary_thetas, detect_thetas])
    all_sample_polar_idxs = torch.hstack(
        [in_mask_sample_polar_idxs, mask_boundary_phi_idxs]
    )
    all_sample_polar_data_idxs = torch.hstack(
        [in_mask_sample_polar_data_idxs, mask_boundary_phi_data_idxs]
    )
    sh_values = toSHValues(
        sh_degree_max,
        sh_params,
        all_sample_phis,
        all_sample_thetas,
        all_sample_polar_idxs,
    )
    sh_points = toSHPoints(
        rotate_vectors,
        positions,
        sample_sh_directions,
        sh_values,
        all_sample_polar_idxs,
        all_sample_polar_data_idxs,
    )

    return sh_points
