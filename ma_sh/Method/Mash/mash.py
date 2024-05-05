from ma_sh.Method.Mash.mash_unit import (
    toParams,
    toPreLoadUniformSamplePolars,
    toPreLoadMaskBoundaryPhiIdxs,
    toPreLoadBaseValues,
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
    mask_boundary_phi_idxs = toPreLoadMaskBoundaryPhiIdxs(
        anchor_num, mask_boundary_sample_num, idx_dtype, device
    )
    mask_boundary_phis, mask_boundary_base_values, sample_base_values = (
        toPreLoadBaseValues(
            anchor_num, mask_boundary_sample_num, mask_degree_max, sample_phis
        )
    )

    return (
        sample_phis,
        sample_thetas,
        mask_boundary_phis,
        mask_boundary_phi_idxs,
        mask_boundary_base_values,
        sample_base_values,
    )
