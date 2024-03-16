import torch
from math import ceil
from tqdm import trange

import mash_cpp

from ma_sh.Method.Mash.mash_unit import (
    toParams,
    toPreLoadUniformSamplePolars,
    toPreLoadMaskBoundaryIdxs,
    toPreLoadBaseValues,
    toPreLoadSHDirections,
)
from ma_sh.Module.timer import Timer


def test():
    anchor_num = 40
    mask_degree_max = 4
    sh_degree_max = 3
    mask_boundary_sample_num = 100
    sample_polar_num = 1000
    sample_point_scale = 0.5
    idx_dtype = torch.int64
    dtype = torch.float64
    device = "cuda:0"

    mask_params, sh_params, rotate_vectors, positions = toParams(
        anchor_num, mask_degree_max, sh_degree_max, dtype, device
    )

    for i in range(anchor_num):
        mask_params.data[i, 0] = i + 10.0

    for i in range(anchor_num):
        sh_params.data[i, 0] = i + 1.0

    for i in range(anchor_num):
        rotate_vectors.data[i, 0] = i

    for i in range(anchor_num):
        positions.data[i, 0] = i

    sample_phis, sample_thetas = toPreLoadUniformSamplePolars(
        sample_polar_num, dtype, device
    )
    mask_boundary_phi_idxs, mask_boundary_phi_data_idxs = toPreLoadMaskBoundaryIdxs(
        anchor_num, mask_boundary_sample_num, idx_dtype, device
    )
    mask_boundary_phis, mask_boundary_base_values, sample_base_values = (
        toPreLoadBaseValues(
            anchor_num,
            mask_boundary_sample_num,
            mask_degree_max,
            sample_phis,
        )
    )
    sample_sh_directions = toPreLoadSHDirections(sample_phis, sample_thetas)

    mask_boundary_thetas = mash_cpp.toMaskBoundaryThetas(
        mask_params, mask_boundary_base_values, mask_boundary_phi_idxs
    )

    in_max_mask_sample_polar_idxs_vec = mash_cpp.toInMaxMaskSamplePolarIdxsVec(
        anchor_num, sample_thetas, mask_boundary_thetas, mask_boundary_phi_idxs
    )

    in_max_mask_sample_polar_idxs = mash_cpp.toInMaxMaskSamplePolarIdxs(
        in_max_mask_sample_polar_idxs_vec
    )

    in_max_mask_sample_polar_data_idxs = torch.hstack(in_max_mask_sample_polar_idxs_vec)

    in_max_mask_sample_phis = sample_phis[in_max_mask_sample_polar_data_idxs]
    in_max_mask_sample_thetas = sample_thetas[in_max_mask_sample_polar_data_idxs]

    in_max_mask_base_values = sample_base_values[:, in_max_mask_sample_polar_data_idxs]

    in_max_mask_thetas = mash_cpp.toInMaxMaskThetas(
        mask_params,
        in_max_mask_base_values,
        in_max_mask_sample_polar_idxs,
        in_max_mask_sample_polar_data_idxs,
    )

    in_mask_sample_polar_mask = in_max_mask_sample_thetas <= in_max_mask_thetas
    in_mask_sample_phis = in_max_mask_sample_phis[in_mask_sample_polar_mask]
    in_mask_sample_polar_idxs = in_max_mask_sample_polar_idxs[in_mask_sample_polar_mask]
    in_mask_sample_polar_data_idxs = in_max_mask_sample_polar_data_idxs[
        in_mask_sample_polar_mask
    ]
    in_mask_base_values = in_max_mask_base_values[:, in_mask_sample_polar_mask]

    in_mask_sample_theta_weights = mash_cpp.toInMaskSampleThetaWeights(
        in_max_mask_sample_thetas, in_max_mask_thetas, in_mask_sample_polar_mask
    )

    detect_thetas = mash_cpp.toDetectThetas(
        mask_params,
        in_mask_base_values,
        in_mask_sample_polar_idxs,
        in_mask_sample_theta_weights,
    )

    in_mask_sh_values = mash_cpp.toSHValues(
        sh_degree_max,
        sh_params,
        in_mask_sample_phis,
        detect_thetas,
        in_mask_sample_polar_idxs,
    )

    in_mask_sh_points = mash_cpp.toSHPoints(
        sh_params,
        rotate_vectors,
        positions,
        sample_sh_directions,
        in_mask_sh_values,
        in_mask_sample_polar_idxs,
        in_mask_sample_polar_data_idxs,
    )

    mask_boundary_sh_values = mash_cpp.toSHValues(
        sh_degree_max,
        sh_params,
        mask_boundary_phis,
        mask_boundary_thetas,
        mask_boundary_phi_idxs,
    )

    mask_boundary_sh_points = mash_cpp.toSHPoints(
        sh_params,
        rotate_vectors,
        positions,
        sample_sh_directions,
        mask_boundary_sh_values,
        mask_boundary_phi_idxs,
        mask_boundary_phi_data_idxs,
    )

    sample_point_num = ceil(in_mask_sh_points.shape[0] * sample_point_scale)

    v_in_mask_sh_points = in_mask_sh_points.reshape(1, -1, 3)

    float_detach_v_in_mask_sh_points = v_in_mask_sh_points.detach().type(torch.float32)

    v_fps_in_mask_sh_point_idxs = mash_cpp.furthest_point_sampling(
        float_detach_v_in_mask_sh_points, sample_point_num
    )

    fps_in_mask_sh_point_idxs = v_fps_in_mask_sh_point_idxs.reshape(-1)

    fps_in_mask_sh_points = in_mask_sh_points[fps_in_mask_sh_point_idxs]

    sh_points = torch.vstack([fps_in_mask_sh_points, mask_boundary_sh_points])

    # ====================== BUG START =========================
    timer = Timer()

    for _ in trange(10):
        print("==== toInMaxMaskSamplePolarIdxsVec")
        detach_mask_boundary_thetas = mask_boundary_thetas.detach()
        print("detach:", timer.now())

        max_values_list = []
        for i in range(anchor_num):
            timer.reset()
            current_data_mask = mask_boundary_phi_idxs == i
            print("itr", i, "get mask:", timer.now())

            # FIXME: when add furthest_point_sampling, first call of this will be too slow!
            timer.reset()
            masked_data = detach_mask_boundary_thetas[current_data_mask]
            print("itr", i, "get masked data:", timer.now())

            timer.reset()
            current_max_value = torch.max(masked_data)
            print("itr", i, "get max:", timer.now())

            timer.reset()
            max_values_list.append(current_max_value)
            print("itr", i, "append:", timer.now())

            if i > 2:
                break

        timer.reset()
        mask_boundary_max_thetas = torch.hstack(max_values_list)
        print("hstack:", timer.now())

        timer.reset()
        in_max_mask_sample_polar_idxs_vec = mash_cpp.toLowerIdxsVec(
            sample_thetas, mask_boundary_max_thetas
        )
        print("toLowerIdxsVec:", timer.now())

        print("==== toInMaxMaskSamplePolarIdxsVec")

        timer.reset()
        v_fps_in_mask_sh_point_idxs = mash_cpp.furthest_point_sampling(
            float_detach_v_in_mask_sh_points, sample_point_num
        )
        print("furthest_point_sampling", timer.now())

    exit()
    return True
