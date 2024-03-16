import torch
import mash_cpp

from ma_sh.Config.constant import EPSILON
from ma_sh.Config.mode import DEBUG
from ma_sh.Method.check import checkFormat


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
    assert checkFormat(sh_params, dtype, device, [anchor_num, (sh_degree_max + 1) ** 2])
    assert checkFormat(rotate_vectors, dtype, device, [anchor_num, 3])
    assert checkFormat(positions, dtype, device, [anchor_num, 3])

    return mask_params, sh_params, rotate_vectors, positions


def toPreLoadUniformSamplePolars(sample_polar_num: int, dtype, device: str):
    sample_phis = mash_cpp.toUniformSamplePhis(sample_polar_num).type(dtype).to(device)
    sample_thetas = (
        mash_cpp.toUniformSampleThetas(sample_polar_num).type(dtype).to(device)
    )

    assert checkFormat(sample_phis, dtype, device, [sample_polar_num], False)
    assert checkFormat(sample_thetas, dtype, device, [sample_polar_num], False)

    return sample_phis, sample_thetas


def toPreLoadMaskBoundaryIdxs(
    anchor_num: int, mask_boundary_sample_num: int, idx_dtype, device: str
):
    mask_boundary_phi_counts = (
        torch.ones(anchor_num).type(idx_dtype).to(device) * mask_boundary_sample_num
    )
    mask_boundary_phi_idxs = mash_cpp.toIdxs(mask_boundary_phi_counts)
    mask_boundary_phi_data_idxs = (
        mash_cpp.toDataIdxs(anchor_num, mask_boundary_sample_num)
        .type(idx_dtype)
        .to(device)
    )

    assert checkFormat(mask_boundary_phi_counts, idx_dtype, device, [anchor_num], False)
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
        mash_cpp.toMaskBoundaryPhis(anchor_num, mask_boundary_sample_num)
        .type(sample_phis.dtype)
        .to(sample_phis.device)
    )
    mask_boundary_base_values = mash_cpp.toMaskBaseValues(
        mask_boundary_phis, mask_degree_max
    )
    sample_base_values = mash_cpp.toMaskBaseValues(sample_phis, mask_degree_max)

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
    sample_sh_directions = mash_cpp.toSHDirections(sample_phis, sample_thetas)

    dtype = sample_phis.dtype
    device = str(sample_phis.device)

    assert checkFormat(
        sample_sh_directions, dtype, device, [sample_phis.shape[0], 3], False
    )

    return sample_sh_directions
