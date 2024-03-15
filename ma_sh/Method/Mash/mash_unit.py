import torch

from ma_sh.Config.constant import EPSILON
from ma_sh.Config.mode import DEBUG
from ma_sh.Method.check import checkFormat
from ma_sh.Method.Mash.kernel_unit import (
    toMaxValues,
    toCounts,
    toIdxs,
    toDataIdxs,
    toLowerIdxsList,
    toMaskBaseValues,
    toRotateMatrixs,
    toUniformSamplePhis,
    toUniformSampleThetas,
    toMaskBoundaryPhis,
    toSHBaseValues,
    toSHDirections,
    toValues,
)


def toParams(
    anchor_num: int, mask_degree_max: int, sh_degree_max: int, dtype, device: str
):
    mask_params = (
        torch.ones([anchor_num, mask_degree_max * 2 + 1]).type(dtype).to(device)
        * EPSILON
    )
    mask_params[:, 0] = 1.0

    sh_params = (
        torch.ones([anchor_num, (sh_degree_max + 1) ** 2]).type(dtype).to(device)
        * EPSILON
    )
    sh_params[:, 0] = 1.0

    rotate_vectors = torch.ones([anchor_num, 3]).type(dtype).to(device) * EPSILON

    positions = torch.ones([anchor_num, 3]).type(dtype).to(device) * EPSILON

    if DEBUG:
        mask_params.requires_grad_(True)
        sh_params.requires_grad_(True)

        for i in range(anchor_num):
            rotate_vectors[i, 0] = i
        rotate_vectors.requires_grad_(True)

        for i in range(anchor_num):
            positions[i, 0] = i
        positions.requires_grad_(True)

        assert checkFormat(
            mask_params, dtype, device, [anchor_num, mask_degree_max * 2 + 1]
        )
        assert checkFormat(
            sh_params, dtype, device, [anchor_num, (sh_degree_max + 1) ** 2]
        )
        assert checkFormat(rotate_vectors, dtype, device, [anchor_num, 3])
        assert checkFormat(positions, dtype, device, [anchor_num, 3])

    return mask_params, sh_params, rotate_vectors, positions


def toPreLoadUniformSamplePolars(sample_polar_num: int, dtype, device: str):
    sample_phis = toUniformSamplePhis(sample_polar_num).type(dtype).to(device)
    sample_thetas = toUniformSampleThetas(sample_polar_num).type(dtype).to(device)

    if DEBUG:
        assert checkFormat(sample_phis, dtype, device, [sample_polar_num], False)
        assert checkFormat(sample_thetas, dtype, device, [sample_polar_num], False)

    return sample_phis, sample_thetas


def toPreLoadMaskBoundaryIdxs(
    anchor_num: int, mask_boundary_sample_num: int, idx_dtype, device: str
):
    mask_boundary_phi_counts = (
        torch.ones(anchor_num).type(idx_dtype).to(device) * mask_boundary_sample_num
    )
    mask_boundary_phi_idxs = toIdxs(mask_boundary_phi_counts)
    mask_boundary_phi_data_idxs = (
        toDataIdxs(anchor_num, mask_boundary_sample_num).type(idx_dtype).to(device)
    )

    if DEBUG:
        assert checkFormat(
            mask_boundary_phi_counts, idx_dtype, device, [anchor_num], False
        )
        assert checkFormat(
            mask_boundary_phi_idxs,
            idx_dtype,
            device,
            [anchor_num * mask_boundary_sample_num],
            False,
        )
        assert checkFormat(
            mask_boundary_phi_data_idxs,
            idx_dtype,
            device,
            [anchor_num * mask_boundary_sample_num],
            False,
        )

    return mask_boundary_phi_idxs, mask_boundary_phi_data_idxs


def toPreLoadBaseValues(
    anchor_num: int,
    mask_boundary_sample_num: int,
    mask_degree_max: int,
    sample_phis: torch.Tensor,
):
    mask_boundary_phis = (
        toMaskBoundaryPhis(anchor_num, mask_boundary_sample_num)
        .type(sample_phis.dtype)
        .to(sample_phis.device)
    )
    mask_boundary_base_values = toMaskBaseValues(mask_boundary_phis, mask_degree_max)
    sample_base_values = toMaskBaseValues(sample_phis, mask_degree_max)

    if DEBUG:
        dtype = sample_phis.dtype
        device = str(sample_phis.device)

        assert checkFormat(
            mask_boundary_phis,
            dtype,
            device,
            [anchor_num * mask_boundary_sample_num],
            False,
        )
        assert checkFormat(
            mask_boundary_base_values,
            dtype,
            device,
            [mask_degree_max * 2 + 1, mask_boundary_phis.shape[0]],
            False,
        )
        assert checkFormat(
            sample_base_values,
            dtype,
            device,
            [mask_degree_max * 2 + 1, sample_phis.shape[0]],
            False,
        )

    return mask_boundary_phis, mask_boundary_base_values, sample_base_values


def toPreLoadSHDirections(sample_phis: torch.Tensor, sample_thetas: torch.Tensor):
    sample_sh_directions = toSHDirections(sample_phis, sample_thetas)

    if DEBUG:
        dtype = sample_phis.dtype
        device = str(sample_phis.device)

        assert checkFormat(
            sample_sh_directions, dtype, device, [sample_phis.shape[0], 3], False
        )

    return sample_sh_directions


def toMaskBoundaryThetas(
    mask_params: torch.Tensor,
    mask_boundary_base_values: torch.Tensor,
    mask_boundary_phi_idxs: torch.Tensor,
):
    mask_boundary_thetas = toValues(
        mask_params, mask_boundary_base_values, mask_boundary_phi_idxs
    )

    if DEBUG:
        dtype = mask_params.dtype
        device = str(mask_params.device)

        assert checkFormat(
            mask_boundary_thetas,
            dtype,
            device,
            [mask_boundary_base_values.shape[1]],
            True,
        )

    return mask_boundary_thetas


def toInMaxMaskIdxs(sample_thetas, mask_boundary_thetas, mask_boundary_phi_idxs):
    mask_boundary_max_thetas = toMaxValues(
        mask_boundary_thetas.detach(), mask_boundary_phi_idxs
    )
    in_max_mask_sample_polar_idxs_list = toLowerIdxsList(
        sample_thetas, mask_boundary_max_thetas
    )
    in_max_mask_sample_polar_counts = toCounts(in_max_mask_sample_polar_idxs_list)
    in_max_mask_sample_polar_idxs = toIdxs(in_max_mask_sample_polar_counts)
    in_max_mask_sample_polar_data_idxs = torch.hstack(
        in_max_mask_sample_polar_idxs_list
    )

    if DEBUG:
        idx_dtype = mask_boundary_phi_idxs.dtype
        dtype = mask_boundary_thetas.dtype
        device = str(mask_boundary_thetas.device)

        assert checkFormat(mask_boundary_max_thetas, dtype, device, None, False)
        assert checkFormat(
            in_max_mask_sample_polar_idxs_list[0], idx_dtype, device, None, False
        )
        assert checkFormat(
            in_max_mask_sample_polar_counts,
            idx_dtype,
            device,
            None,
            False,
        )
        assert checkFormat(
            in_max_mask_sample_polar_idxs,
            idx_dtype,
            device,
            [torch.sum(in_max_mask_sample_polar_counts)],
            False,
        )

    return in_max_mask_sample_polar_idxs, in_max_mask_sample_polar_data_idxs


def toInMaxMaskPolars(sample_phis, sample_thetas, in_max_mask_sample_polar_data_idxs):
    in_max_mask_sample_phis = sample_phis[in_max_mask_sample_polar_data_idxs]
    in_max_mask_sample_thetas = sample_thetas[in_max_mask_sample_polar_data_idxs]

    if DEBUG:
        dtype = sample_phis.dtype
        device = str(sample_phis.device)

        assert checkFormat(
            in_max_mask_sample_phis,
            dtype,
            device,
            [in_max_mask_sample_polar_data_idxs.shape[0]],
            False,
        )
        assert checkFormat(
            in_max_mask_sample_thetas,
            dtype,
            device,
            [in_max_mask_sample_polar_data_idxs.shape[0]],
            False,
        )

    return in_max_mask_sample_phis, in_max_mask_sample_thetas


def toInMaxMaskBaseValues(
    sample_base_values: torch.Tensor, in_max_mask_sample_polar_data_idxs: torch.Tensor
) -> torch.Tensor:
    in_max_mask_base_values = sample_base_values[:, in_max_mask_sample_polar_data_idxs]

    if DEBUG:
        dtype = sample_base_values.dtype
        device = str(sample_base_values.device)

        assert checkFormat(
            in_max_mask_base_values,
            dtype,
            device,
            [sample_base_values.shape[0], in_max_mask_sample_polar_data_idxs.shape[0]],
            False,
        )

    return in_max_mask_base_values


def toInMaxMaskThetas(
    mask_params: torch.Tensor,
    in_max_mask_base_values: torch.Tensor,
    in_max_mask_sample_polar_idxs: torch.Tensor,
):
    with torch.no_grad():
        in_max_mask_thetas = toValues(
            mask_params, in_max_mask_base_values, in_max_mask_sample_polar_idxs
        )

    if DEBUG:
        dtype = mask_params.dtype
        device = str(mask_params.device)

        assert checkFormat(
            in_max_mask_thetas,
            dtype,
            device,
            [in_max_mask_sample_polar_idxs.shape[0]],
            False,
        )

    return in_max_mask_thetas


def toInMaskSamplePolarWeights(
    mask_params,
    in_max_mask_base_values,
    in_max_mask_thetas,
    in_max_mask_sample_phis,
    in_max_mask_sample_thetas,
    in_max_mask_sample_polar_idxs,
    in_max_mask_sample_polar_data_idxs,
):
    in_mask_sample_polar_mask = in_max_mask_sample_thetas <= in_max_mask_thetas
    in_mask_sample_phis = in_max_mask_sample_phis[in_mask_sample_polar_mask]
    in_mask_sample_thetas = in_max_mask_sample_thetas[in_mask_sample_polar_mask]
    in_mask_sample_polar_idxs = in_max_mask_sample_polar_idxs[in_mask_sample_polar_mask]
    in_mask_sample_polar_data_idxs = in_max_mask_sample_polar_data_idxs[
        in_mask_sample_polar_mask
    ]
    in_mask_base_values = in_max_mask_base_values[:, in_mask_sample_polar_mask]
    in_mask_thetas = in_max_mask_thetas[in_mask_sample_polar_mask]
    in_mask_sample_theta_weights = in_mask_sample_thetas / in_mask_thetas

    if DEBUG:
        dtype = mask_params.dtype
        device = str(mask_params.device)

        assert checkFormat(in_mask_sample_phis, dtype, device, None, False)
        assert checkFormat(
            in_mask_sample_thetas, dtype, device, [in_mask_sample_phis.shape[0]], False
        )
        assert checkFormat(
            in_mask_sample_polar_idxs,
            in_max_mask_sample_polar_idxs.dtype,
            device,
            [in_mask_sample_phis.shape[0]],
            False,
        )
        assert checkFormat(
            in_mask_sample_polar_data_idxs,
            in_max_mask_sample_polar_data_idxs.dtype,
            device,
            [in_mask_sample_phis.shape[0]],
            False,
        )

    return (
        in_mask_sample_phis,
        in_mask_sample_polar_idxs,
        in_mask_sample_polar_data_idxs,
        in_mask_base_values,
        in_mask_sample_theta_weights,
    )


def toSamplePolars(
    mask_params,
    in_mask_base_values,
    in_mask_sample_polar_idxs,
    in_mask_sample_theta_weights,
):
    detect_boundary_thetas = toValues(
        mask_params, in_mask_base_values, in_mask_sample_polar_idxs
    )
    detect_thetas = in_mask_sample_theta_weights * detect_boundary_thetas

    if DEBUG:
        dtype = mask_params.dtype
        device = str(mask_params.device)

        assert checkFormat(
            detect_boundary_thetas,
            dtype,
            device,
            [in_mask_sample_polar_idxs.shape[0]],
            True,
        )
        assert checkFormat(
            detect_thetas, dtype, device, [in_mask_sample_polar_idxs.shape[0]], True
        )

    return detect_thetas


def toSHValues(
    sh_degree_max,
    sh_params,
    in_mask_sample_phis,
    detect_thetas,
    in_mask_sample_polar_idxs,
):
    sh_base_values = toSHBaseValues(in_mask_sample_phis, detect_thetas, sh_degree_max)
    sh_values = toValues(sh_params, sh_base_values, in_mask_sample_polar_idxs)

    if DEBUG:
        dtype = sh_params.dtype
        device = str(sh_params.device)

        assert checkFormat(
            sh_base_values,
            dtype,
            device,
            [(sh_degree_max + 1) ** 2, in_mask_sample_phis.shape[0]],
            True,
        )
        assert checkFormat(
            sh_values, dtype, device, [in_mask_sample_phis.shape[0]], True
        )

    return sh_values


def toSHPoints(
    rotate_vectors,
    positions,
    sample_sh_directions,
    sh_values,
    in_mask_sample_polar_idxs,
    in_mask_sample_polar_data_idxs,
):
    v_sh_values = sh_values.reshape(-1, 1)
    in_mask_sh_directions = sample_sh_directions[in_mask_sample_polar_data_idxs]
    sh_local_points = v_sh_values * in_mask_sh_directions
    in_mask_rotate_vectors = rotate_vectors[in_mask_sample_polar_idxs]
    in_mask_rotate_matrixs = toRotateMatrixs(in_mask_rotate_vectors)
    v_sh_local_points = sh_local_points.reshape(-1, 1, 3)
    v_sh_local_rotate_points = torch.matmul(v_sh_local_points, in_mask_rotate_matrixs)
    sh_local_rotate_points = v_sh_local_rotate_points.reshape(-1, 3)
    in_mask_positions = positions[in_mask_sample_polar_idxs]
    sh_points = in_mask_positions + sh_local_rotate_points

    if DEBUG:
        dtype = rotate_vectors.dtype
        device = str(rotate_vectors.device)

        assert checkFormat(
            in_mask_sh_directions,
            dtype,
            device,
            [in_mask_sample_polar_data_idxs.shape[0], 3],
            False,
        )
        assert checkFormat(
            sh_local_points, dtype, device, [sh_values.shape[0], 3], True
        )
        assert checkFormat(
            in_mask_rotate_matrixs,
            dtype,
            device,
            [in_mask_rotate_vectors.shape[0], 3, 3],
            True,
        )
        assert checkFormat(
            in_mask_positions, dtype, device, [sh_values.shape[0], 3], True
        )
        assert checkFormat(sh_points, dtype, device, [sh_values.shape[0], 3], True)

    return sh_points
