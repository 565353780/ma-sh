import torch
from tqdm import trange

from ma_sh.Method.torch_thread import setThread
from ma_sh.Method.check import checkFormat
from ma_sh.Method.kernel import (
    toMaxValues,
    toCounts,
    toIdxs,
    toLowerIdxsList,
    toMaskBaseValues,
    toRotateVectors,
    toRotateMatrixs,
    toUniformSamplePhis,
    toUniformSampleThetas,
    toMaskBoundaryPhis,
    toSHBaseValues,
    toValues,
)

from ma_sh.Method.rotate import toRotateMatrixs


def testPreLoadUniformSample(sample_polar_num, dtype=torch.float32, device="cpu"):
    sample_phis = toUniformSamplePhis(sample_polar_num).type(dtype).to(device)
    sample_thetas = toUniformSampleThetas(sample_polar_num).type(dtype).to(device)
    return


def testPreLoadMaskBoundary(
    anchor_num, mask_boundary_sample_num, idx_dtype, dtype, device
):
    mask_boundary_sample_nums = (
        torch.ones(anchor_num).type(idx_dtype).to(device) * mask_boundary_sample_num
    )
    mask_boundary_phi_idxs = toIdxs(mask_boundary_sample_nums)
    mask_boundary_phis = (
        toMaskBoundaryPhis(anchor_num, mask_boundary_sample_num).type(dtype).to(device)
    )
    return


def testMaskBoundary(
    mask_degree_max, mask_params, mask_boundary_phis, mask_boundary_phi_idxs
):
    mask_boundary_base_values = toMaskBaseValues(mask_boundary_phis, mask_degree_max)
    with torch.no_grad():
        mask_boundary_thetas = toValues(
            mask_params, mask_boundary_base_values, mask_boundary_phi_idxs
        )
    return


def testInMaxMaskSamplePolars(
    sample_phis, sample_thetas, mask_boundary_thetas, mask_boundary_phi_idxs
):
    mask_boundary_max_thetas = toMaxValues(mask_boundary_thetas, mask_boundary_phi_idxs)
    in_max_mask_sample_polar_idxs_list = toLowerIdxsList(
        sample_thetas, mask_boundary_max_thetas
    )
    in_max_mask_sample_polar_counts = toCounts(in_max_mask_sample_polar_idxs_list)
    in_max_mask_sample_polar_idxs = toIdxs(in_max_mask_sample_polar_counts)
    in_max_mask_sample_polar_data_idxs = torch.hstack(
        in_max_mask_sample_polar_idxs_list
    )
    in_max_mask_sample_phis = sample_phis[in_max_mask_sample_polar_data_idxs]
    in_max_mask_sample_thetas = sample_thetas[in_max_mask_sample_polar_data_idxs]
    return


def testInMaskSamplePolarWeights(
    mask_degree_max,
    mask_params,
    in_max_mask_sample_phis,
    in_max_mask_sample_thetas,
    in_max_mask_sample_polar_idxs,
):
    in_max_mask_base_values = toMaskBaseValues(in_max_mask_sample_phis, mask_degree_max)
    with torch.no_grad():
        in_max_mask_thetas = toValues(
            mask_params, in_max_mask_base_values, in_max_mask_sample_polar_idxs
        )
    in_mask_sample_polar_mask = in_max_mask_sample_thetas <= in_max_mask_thetas
    in_mask_sample_phis = in_max_mask_sample_phis[in_mask_sample_polar_mask]
    in_mask_sample_thetas = in_max_mask_sample_thetas[in_mask_sample_polar_mask]
    in_mask_sample_polar_idxs = in_max_mask_sample_polar_idxs[in_mask_sample_polar_mask]
    in_mask_base_values = in_max_mask_base_values[:, in_mask_sample_polar_mask]
    in_mask_thetas = in_max_mask_thetas[in_mask_sample_polar_mask]
    in_mask_sample_theta_weights = in_mask_sample_thetas / in_mask_thetas
    return

def testSamplePolars(mask_params, in_mask_base_values, in_mask_sample_polar_idxs, in_mask_sample_theta_weights):
    detect_boundary_thetas = toValues(mask_params, in_mask_base_values, in_mask_sample_polar_idxs)
    detect_thetas = in_mask_sample_theta_weights * detect_boundary_thetas
    return

def testSHValues(
    sh_params,
    in_mask_sample_phis,
    in_mask_sample_thetas,
    sh_degree_max,
    in_mask_sample_polar_idxs,
):
    sh_base_values = toSHBaseValues(
        in_mask_sample_phis, in_mask_sample_thetas, sh_degree_max
    )
    sh_values = toValues(sh_params, sh_base_values, in_mask_sample_polar_idxs)
    return


def test():
    # setThread()

    sample_polar_num = 10
    anchor_num = 4
    mask_degree_max = 5
    mask_boundary_sample_num = 10
    sh_degree_max = 3
    idx_dtype = torch.int64
    dtype = torch.float64
    device = "cpu"

    # Params
    mask_params = (
        torch.zeros([anchor_num, mask_degree_max * 2 + 1]).type(dtype).to(device)
    )
    assert checkFormat(
        mask_params, dtype, device, [anchor_num, mask_degree_max * 2 + 1]
    )

    sh_params = (
        torch.zeros([anchor_num, (sh_degree_max + 1) ** 2]).type(dtype).to(device)
    )
    assert checkFormat(sh_params, dtype, device, [anchor_num, (sh_degree_max + 1) ** 2])

    rotate_vectors = torch.zeros([anchor_num, 3]).type(dtype).to(device)
    assert checkFormat(rotate_vectors, dtype, device, [anchor_num, 3])

    positions = torch.zeros([anchor_num, 3]).type(dtype).to(device)
    assert checkFormat(positions, dtype, device, [anchor_num, 3])

    for i in range(anchor_num):
        mask_params[i, 0] = i + 1.0
    mask_params.requires_grad_(True)

    for i in range(anchor_num):
        sh_params[i, 0] = i + 1.0
    sh_params.requires_grad_(True)

    for i in range(anchor_num):
        rotate_vectors[i, 0] = i
    rotate_vectors.requires_grad_(True)

    for i in range(anchor_num):
        positions[i, 0] = i
    positions.requires_grad_(True)

    # Pre Load Uniform Sample
    sample_phis = toUniformSamplePhis(sample_polar_num).type(dtype).to(device)
    assert checkFormat(sample_phis, dtype, device, [sample_polar_num], False)

    sample_thetas = toUniformSampleThetas(sample_polar_num).type(dtype).to(device)
    assert checkFormat(sample_thetas, dtype, device, [sample_polar_num], False)

    # Pre Load Mask Boundary
    mask_boundary_phi_counts = (
        torch.ones(anchor_num).type(idx_dtype).to(device) * mask_boundary_sample_num
    )
    assert checkFormat(mask_boundary_phi_counts, idx_dtype, device, [anchor_num], False)

    mask_boundary_phi_idxs = toIdxs(mask_boundary_phi_counts)
    assert checkFormat(
        mask_boundary_phi_idxs,
        idx_dtype,
        device,
        [anchor_num * mask_boundary_sample_num],
        False
    )

    mask_boundary_phis = (
        toMaskBoundaryPhis(anchor_num, mask_boundary_sample_num).type(dtype).to(device)
    )
    assert checkFormat(
        mask_boundary_phis, dtype, device, [anchor_num * mask_boundary_sample_num], False
    )

    # Pre Load Base Values
    mask_boundary_base_values = toMaskBaseValues(mask_boundary_phis, mask_degree_max)
    assert checkFormat(
        mask_boundary_base_values,
        dtype,
        device,
        [mask_degree_max * 2 + 1, anchor_num * mask_boundary_sample_num], False
    )

    sample_base_values = toMaskBaseValues(sample_phis, mask_degree_max)
    assert checkFormat(
        sample_base_values,
        dtype,
        device,
        [mask_degree_max * 2 + 1, sample_polar_num], False
    )

    # Pre Load Rotate Matrixs
    rotate_matrixs = toRotateMatrixs(rotate_vectors)
    assert checkFormat(
        rotate_matrixs,
        dtype,
        device,
        [anchor_num, 3, 3], True
    )

    # Pre Load SH Directions
    sample_sh_direction_x = torch.cos(sample_phis) * torch.sin(sample_thetas)
    sample_sh_direction_y = torch.sin(sample_phis) * torch.sin(sample_thetas)
    sample_sh_direction_z = torch.cos(sample_thetas)
    sample_sh_directions = torch.vstack([sample_sh_direction_x, sample_sh_direction_y, sample_sh_direction_z]).transpose(1, 0)
    assert checkFormat(
        sample_sh_directions,
        dtype,
        device,
        [sample_polar_num, 3], False
    )

    # Mask Boundary
    with torch.no_grad():
        mask_boundary_thetas = toValues(
            mask_params, mask_boundary_base_values, mask_boundary_phi_idxs
        )
    assert checkFormat(
        mask_boundary_thetas, dtype, device, [anchor_num * mask_boundary_sample_num], False
    )

    # In Max Mask Sample Polars
    mask_boundary_max_thetas = toMaxValues(mask_boundary_thetas, mask_boundary_phi_idxs)
    assert checkFormat(mask_boundary_max_thetas, dtype, device, [anchor_num], False)

    in_max_mask_sample_polar_idxs_list = toLowerIdxsList(
        sample_thetas, mask_boundary_max_thetas
    )
    # assert checkFormat(in_max_mask_sample_polar_idxs_list[0], idx_dtype, device, None, False)

    in_max_mask_sample_polar_counts = toCounts(in_max_mask_sample_polar_idxs_list)
    assert checkFormat(in_max_mask_sample_polar_counts, in_max_mask_sample_polar_idxs_list[0].dtype, device, [anchor_num], False)

    in_max_mask_sample_polar_idxs = toIdxs(in_max_mask_sample_polar_counts)
    assert checkFormat(
        in_max_mask_sample_polar_idxs,
        in_max_mask_sample_polar_counts.dtype,
        device,
        [torch.sum(in_max_mask_sample_polar_counts)], False
    )

    in_max_mask_sample_polar_data_idxs = torch.hstack(
        in_max_mask_sample_polar_idxs_list
    )

    in_max_mask_sample_phis = sample_phis[in_max_mask_sample_polar_data_idxs]
    assert checkFormat(
        in_max_mask_sample_phis,
        dtype,
        device,
        [in_max_mask_sample_polar_idxs.shape[0]], False
    )

    in_max_mask_sample_thetas = sample_thetas[in_max_mask_sample_polar_data_idxs]
    assert checkFormat(
        in_max_mask_sample_thetas,
        dtype,
        device,
        [in_max_mask_sample_polar_idxs.shape[0]], False
    )

    # In Mask Sample Polar Weights
    in_max_mask_base_values = sample_base_values[:, in_max_mask_sample_polar_data_idxs]
    assert checkFormat(
        in_max_mask_base_values,
        dtype,
        device,
        [mask_degree_max * 2 + 1, in_max_mask_sample_phis.shape[0]], False
    )

    with torch.no_grad():
        in_max_mask_thetas = toValues(
            mask_params, in_max_mask_base_values, in_max_mask_sample_polar_idxs
        )
    assert checkFormat(
        in_max_mask_thetas, dtype, device, [in_max_mask_sample_polar_idxs.shape[0]], False
    )

    in_mask_sample_polar_mask = in_max_mask_sample_thetas <= in_max_mask_thetas

    in_mask_sample_phis = in_max_mask_sample_phis[in_mask_sample_polar_mask]
    assert checkFormat(in_mask_sample_phis, dtype, device, None, False)

    in_mask_sample_thetas = in_max_mask_sample_thetas[in_mask_sample_polar_mask]
    assert checkFormat(
        in_mask_sample_thetas, dtype, device, [in_mask_sample_phis.shape[0]], False
    )

    in_mask_sample_polar_idxs = in_max_mask_sample_polar_idxs[in_mask_sample_polar_mask]
    assert checkFormat(
        in_mask_sample_polar_idxs, in_max_mask_sample_polar_idxs.dtype, device, [in_mask_sample_phis.shape[0]], False
    )

    in_mask_sample_polar_data_idxs = in_max_mask_sample_polar_data_idxs[in_mask_sample_polar_mask]
    assert checkFormat(
        in_mask_sample_polar_data_idxs, in_max_mask_sample_polar_data_idxs.dtype, device, [in_mask_sample_phis.shape[0]], False
    )

    in_mask_base_values = in_max_mask_base_values[:, in_mask_sample_polar_mask]

    in_mask_thetas = in_max_mask_thetas[in_mask_sample_polar_mask]

    in_mask_sample_theta_weights = in_mask_sample_thetas / in_mask_thetas

    in_mask_sh_directions = sample_sh_directions[in_mask_sample_polar_data_idxs]
    assert checkFormat(
        in_mask_sh_directions, dtype, device, [in_mask_sample_polar_data_idxs.shape[0], 3], False
    )

    # Sample Polars
    detect_boundary_thetas = toValues(mask_params, in_mask_base_values, in_mask_sample_polar_idxs)
    assert checkFormat(
        detect_boundary_thetas, dtype, device, [in_mask_sample_polar_idxs.shape[0]], True
    )

    detect_thetas = in_mask_sample_theta_weights * detect_boundary_thetas
    assert checkFormat(
        detect_thetas, dtype, device, [in_mask_sample_polar_idxs.shape[0]], True
    )

    # SH Values
    sh_base_values = toSHBaseValues(
        in_mask_sample_phis, detect_thetas, sh_degree_max
    )
    assert checkFormat(
        sh_base_values,
        dtype,
        device,
        [(sh_degree_max + 1) ** 2, in_mask_sample_phis.shape[0]], True
    )

    sh_values = toValues(sh_params, sh_base_values, in_mask_sample_polar_idxs)
    assert checkFormat(
        sh_values,
        dtype,
        device,
        [in_mask_sample_phis.shape[0]], True
    )

    # SH Points

    v_sh_values = sh_values.reshape(-1, 1)

    sh_local_points = v_sh_values * in_mask_sh_directions
    assert checkFormat(
        sh_local_points,
        dtype,
        device,
        [sh_values.shape[0], 3], True
    )

    in_mask_positions = positions[in_mask_sample_polar_idxs]
    assert checkFormat(
        in_mask_positions,
        dtype,
        device,
        [sh_values.shape[0], 3], True
    )

    sh_points = in_mask_positions + sh_local_points
    assert checkFormat(
        sh_points,
        dtype,
        device,
        [sh_values.shape[0], 3], True
    )

    # Speed
    test_num = 1000

    print("[INFO][speed::test]")
    print("\t start speed test...")
    print("\t testPreLoadUniformSample")
    for _ in trange(test_num):
        testPreLoadUniformSample(sample_polar_num, dtype, device)

    print("\t testPreLoadMaskBoundary")
    for _ in trange(test_num):
        testPreLoadMaskBoundary(
            anchor_num, mask_boundary_sample_num, idx_dtype, dtype, device
        )

    print("\t testMaskBoundary")
    for _ in trange(test_num):
        testMaskBoundary(
            mask_degree_max,
            mask_params,
            mask_boundary_phis,
            mask_boundary_phi_idxs,
        )

    print("\t testInMaxMaskSamplePolars")
    for _ in trange(test_num):
        testInMaxMaskSamplePolars(
            sample_phis,
            sample_thetas,
            mask_boundary_thetas,
            mask_boundary_phi_idxs,
        )

    print("\t testInMaskSamplePolarWeights")
    for _ in trange(test_num):
        testInMaskSamplePolarWeights(
            mask_degree_max,
            mask_params,
            in_max_mask_sample_phis,
            in_max_mask_sample_thetas,
            in_max_mask_sample_polar_idxs,
        )

    print("\t testSamplePolars")
    for _ in trange(test_num):
        testSamplePolars(
            mask_params,
            in_mask_base_values,
            in_mask_sample_polar_idxs,
            in_mask_sample_theta_weights,
        )

    print("\t testSHValues")
    for _ in trange(test_num):
        testSHValues(
            sh_params,
            in_mask_sample_phis,
            in_mask_sample_thetas,
            sh_degree_max,
            in_mask_sample_polar_idxs,
        )
    return True
