import torch
from tqdm import trange

from ma_sh.Method.mash_unit import (
    toParams,
    toPreLoadUniformSamplePolars,
    toPreLoadMaskBoundaryIdxs,
    toPreLoadBaseValues,
    toPreLoadRotateMatrixs,
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

from ma_sh.Method.torch_thread import setThread


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
        anchor_num, mask_boundary_sample_num, mask_degree_max, sample_phis
    )
    rotate_matrixs = toPreLoadRotateMatrixs(rotate_vectors)
    # rotate_matrixs.retain_grad()

    sample_sh_directions = toPreLoadSHDirections(sample_phis, sample_thetas)

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
        rotate_matrixs,
        positions,
        sample_sh_directions,
        sh_values,
        in_mask_sample_polar_idxs,
        in_mask_sample_polar_data_idxs,
    )

    # rotate_matrixs.grad.zero_()

    # Speed
    test_num = 1000

    print("[INFO][speed::test]")
    print("\t start speed test...")

    print("\t toParams")
    for _ in trange(test_num):
        mask_params, sh_params, rotate_vectors, positions = toParams(
            anchor_num, mask_degree_max, sh_degree_max, dtype, device
        )

    print("\t toPreLoadUniformSamplePolars")
    for _ in trange(test_num):
        sample_phis, sample_thetas = toPreLoadUniformSamplePolars(
            sample_polar_num, dtype, device
        )

    print("\t toPreLoadMaskBoundary")
    for _ in trange(test_num):
        mask_boundary_phi_idxs = toPreLoadMaskBoundaryIdxs(
            anchor_num, mask_boundary_sample_num, idx_dtype, device
        )

    print("\t toPreLoadBaseValues")
    for _ in trange(test_num):
        mask_boundary_base_values, sample_base_values = toPreLoadBaseValues(
            anchor_num, mask_boundary_sample_num, mask_degree_max, sample_phis
        )

    print("\t toPreLoadSHDirections")
    for _ in trange(test_num):
        sample_sh_directions = toPreLoadSHDirections(sample_phis, sample_thetas)

    print("\t toPreLoadRotateMatrixs")
    for _ in trange(test_num):
        rotate_matrixs = toPreLoadRotateMatrixs(rotate_vectors)
        # rotate_matrixs.retain_grad()

    print("\t toMaskBoundaryThetas")
    for _ in trange(test_num):
        mask_boundary_thetas = toMaskBoundaryThetas(
            mask_params, mask_boundary_base_values, mask_boundary_phi_idxs
        )

    print("\t toInMaxMaskIdxs")
    for _ in trange(test_num):
        (
            in_max_mask_sample_polar_idxs,
            in_max_mask_sample_polar_data_idxs,
        ) = toInMaxMaskIdxs(sample_thetas, mask_boundary_thetas, mask_boundary_phi_idxs)

    print("\t toInMaxMaskPolars")
    for _ in trange(test_num):
        (
            in_max_mask_sample_phis,
            in_max_mask_sample_thetas,
        ) = toInMaxMaskPolars(
            sample_phis, sample_thetas, in_max_mask_sample_polar_data_idxs
        )

    print("\t toInMaxMaskThetas")
    for _ in trange(test_num):
        (
            in_max_mask_base_values,
            in_max_mask_thetas,
        ) = toInMaxMaskThetas(
            mask_params,
            sample_base_values,
            in_max_mask_sample_polar_idxs,
            in_max_mask_sample_polar_data_idxs,
        )

    print("\t toInMaskSamplePolarWeights")
    for _ in trange(test_num):
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

    print("\t toSamplePolars")
    for _ in trange(test_num):
        detect_thetas = toSamplePolars(
            mask_params,
            in_mask_base_values,
            in_mask_sample_polar_idxs,
            in_mask_sample_theta_weights,
        )

    print("\t toSHValues")
    for _ in trange(test_num):
        sh_values = toSHValues(
            sh_degree_max,
            sh_params,
            in_mask_sample_phis,
            detect_thetas,
            in_mask_sample_polar_idxs,
        )

    print("\t toSHPoints")
    for _ in trange(test_num):
        sh_points = toSHPoints(
            rotate_matrixs,
            positions,
            sample_sh_directions,
            sh_values,
            in_mask_sample_polar_idxs,
            in_mask_sample_polar_data_idxs,
        )

    return True
