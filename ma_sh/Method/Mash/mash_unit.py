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
    mask_params[:, 0] = -4.0

    sh_params = (
        torch.ones([anchor_num, (sh_degree_max + 1) ** 2]).type(dtype).to(device)
        * EPSILON
    )
    sh_params[:, 0] = 0.1

    rotate_vectors = torch.ones([anchor_num, 3]).type(dtype).to(device) * EPSILON

    positions = torch.ones([anchor_num, 3]).type(dtype).to(device) * EPSILON

    if DEBUG:
        mask_params.requires_grad_(True)
        sh_params.requires_grad_(True)

        for i in range(anchor_num):
            rotate_vectors[i, 0] = i + 1
        rotate_vectors.requires_grad_(True)

        for i in range(anchor_num):
            positions[i, 0] = i + 1
        positions.requires_grad_(True)

    assert checkFormat(
        mask_params, dtype, device, [anchor_num, mask_degree_max * 2 + 1]
    )
    assert checkFormat(sh_params, dtype, device, [anchor_num, (sh_degree_max + 1) ** 2])
    assert checkFormat(rotate_vectors, dtype, device, [anchor_num, 3])
    assert checkFormat(positions, dtype, device, [anchor_num, 3])

    return mask_params, sh_params, rotate_vectors, positions


def toPreLoadMaskBoundaryPhiIdxs(
    anchor_num: int, mask_boundary_sample_num: int, idx_dtype, device: str
):
    mask_boundary_phi_counts = (
        torch.ones(anchor_num).type(idx_dtype).to(device) * mask_boundary_sample_num
    )
    mask_boundary_phi_idxs = mash_cpp.toIdxs(mask_boundary_phi_counts)

    assert checkFormat(mask_boundary_phi_counts, idx_dtype, device, [anchor_num], False)
    assert checkFormat(
        mask_boundary_phi_idxs,
        idx_dtype,
        device,
        [anchor_num * mask_boundary_sample_num],
        False,
    )

    return mask_boundary_phi_idxs


def toPreLoadBaseValues(
    anchor_num: int,
    mask_boundary_sample_num: int,
    mask_degree_max: int,
    dtype,
    device: str,
):
    mask_boundary_phis = (
        mash_cpp.toMaskBoundaryPhis(anchor_num, mask_boundary_sample_num)
        .type(dtype)
        .to(device)
    )
    mask_boundary_base_values = mash_cpp.toMaskBaseValues(
        mask_boundary_phis, mask_degree_max
    )

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

    return mask_boundary_phis, mask_boundary_base_values
