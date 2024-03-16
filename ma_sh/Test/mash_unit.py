import torch
from math import ceil
from tqdm import trange

import pointnet2_ops
import mash_cpp

from ma_sh.Method.check import checkFormat
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

    for _ in trange(10):
        timer = Timer()
        mask_boundary_thetas = mash_cpp.toMaskBoundaryThetas(
            mask_params, mask_boundary_base_values, mask_boundary_phi_idxs
        )
        assert checkFormat(
            mask_boundary_thetas,
            dtype,
            device,
            [mask_boundary_base_values.shape[1]],
            True,
        )
        now = timer.now()
        print("toMaskBoundaryThetas:", now)
        timer.reset()

        """
        in_max_mask_sample_polar_idxs_vec = mash_cpp.toInMaxMaskSamplePolarIdxsVec(
            anchor_num, sample_thetas, mask_boundary_thetas, mask_boundary_phi_idxs
        )
        assert checkFormat(
            in_max_mask_sample_polar_idxs_vec[0],
            idx_dtype,
            device,
            None,
            False,
        )
        now = timer.now()
        print("toInMaxMaskSamplePolarIdxsVec:", now)
        timer.reset()
        """

        print("==== toInMaxMaskSamplePolarIdxsVec")
        detach_mask_boundary_thetas = mask_boundary_thetas.detach()
        now = timer.now()
        print("detach:", now)
        timer.reset()

        """
        mask_boundary_max_thetas = mash_cpp.toMaxValues(
            anchor_num, detach_mask_boundary_thetas, mask_boundary_phi_idxs
        )
        now = timer.now()
        print("toMaxValues:", now)
        timer.reset()
        """

        max_values_list = []
        for i in range(anchor_num):
            current_data_mask = mask_boundary_phi_idxs == i
            now = timer.now()
            print("itr", i, "get mask:", now)
            timer.reset()

            # FIXME: when add furthest_point_sampling, first call of this will be too slow!
            masked_data = detach_mask_boundary_thetas[current_data_mask]
            now = timer.now()
            print("itr", i, "get masked data:", now)
            timer.reset()

            current_max_value = torch.max(masked_data)
            now = timer.now()
            print("itr", i, "get max:", now)
            timer.reset()

            max_values_list.append(current_max_value)
            now = timer.now()
            print("itr", i, "append:", now)
            timer.reset()

        mask_boundary_max_thetas = torch.hstack(max_values_list)
        now = timer.now()
        print("hstack:", now)
        timer.reset()

        in_max_mask_sample_polar_idxs_vec = mash_cpp.toLowerIdxsVec(
            sample_thetas, mask_boundary_max_thetas
        )
        now = timer.now()
        print("toLowerIdxsVec:", now)
        timer.reset()

        print("==== toInMaxMaskSamplePolarIdxsVec")

        in_max_mask_sample_polar_idxs = mash_cpp.toInMaxMaskSamplePolarIdxs(
            in_max_mask_sample_polar_idxs_vec
        )
        sample_polar_num = 0
        for sample_polar_idxs in in_max_mask_sample_polar_idxs_vec:
            sample_polar_num += sample_polar_idxs.shape[0]
        assert checkFormat(
            in_max_mask_sample_polar_idxs,
            idx_dtype,
            device,
            [sample_polar_num],
            False,
        )
        now = timer.now()
        print("toInMaxMaskSamplePolarIdxs:", now)
        timer.reset()

        in_max_mask_sample_polar_data_idxs = torch.hstack(
            in_max_mask_sample_polar_idxs_vec
        )
        assert checkFormat(
            in_max_mask_sample_polar_data_idxs,
            idx_dtype,
            device,
            [sample_polar_num],
            False,
        )

        in_max_mask_sample_phis = sample_phis[in_max_mask_sample_polar_data_idxs]
        in_max_mask_sample_thetas = sample_thetas[in_max_mask_sample_polar_data_idxs]
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

        in_max_mask_base_values = sample_base_values[
            :, in_max_mask_sample_polar_data_idxs
        ]
        assert checkFormat(
            in_max_mask_base_values,
            dtype,
            device,
            [
                sample_base_values.shape[0],
                in_max_mask_sample_polar_data_idxs.shape[0],
            ],
            False,
        )
        now = timer.now()
        print("index1:", now)
        timer.reset()

        in_max_mask_thetas = mash_cpp.toInMaxMaskThetas(
            mask_params,
            in_max_mask_base_values,
            in_max_mask_sample_polar_idxs,
            in_max_mask_sample_polar_data_idxs,
        )
        assert checkFormat(
            in_max_mask_thetas,
            dtype,
            device,
            [in_max_mask_sample_polar_idxs.shape[0]],
            False,
        )
        now = timer.now()
        print("toInMaxMaskThetas:", now)
        timer.reset()

        in_mask_sample_polar_mask = in_max_mask_sample_thetas <= in_max_mask_thetas
        in_mask_sample_phis = in_max_mask_sample_phis[in_mask_sample_polar_mask]
        in_mask_sample_polar_idxs = in_max_mask_sample_polar_idxs[
            in_mask_sample_polar_mask
        ]
        in_mask_sample_polar_data_idxs = in_max_mask_sample_polar_data_idxs[
            in_mask_sample_polar_mask
        ]
        in_mask_base_values = in_max_mask_base_values[:, in_mask_sample_polar_mask]
        assert checkFormat(in_mask_sample_phis, dtype, device, None, False)
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
        now = timer.now()
        print("index2:", now)
        timer.reset()

        in_mask_sample_theta_weights = mash_cpp.toInMaskSampleThetaWeights(
            in_max_mask_sample_thetas, in_max_mask_thetas, in_mask_sample_polar_mask
        )
        assert checkFormat(
            in_mask_sample_theta_weights,
            dtype,
            device,
            [in_mask_sample_phis.shape[0]],
            False,
        )
        now = timer.now()
        print("toInMaskSampleThetaWeights:", now)
        timer.reset()

        detect_thetas = mash_cpp.toDetectThetas(
            mask_params,
            in_mask_base_values,
            in_mask_sample_polar_idxs,
            in_mask_sample_theta_weights,
        )
        assert checkFormat(
            detect_thetas, dtype, device, [in_mask_sample_polar_idxs.shape[0]], True
        )
        now = timer.now()
        print("toDetectThetas:", now)
        timer.reset()

        in_mask_sh_values = mash_cpp.toSHValues(
            sh_degree_max,
            sh_params,
            in_mask_sample_phis,
            detect_thetas,
            in_mask_sample_polar_idxs,
        )
        assert checkFormat(
            in_mask_sh_values, dtype, device, [in_mask_sample_phis.shape[0]], True
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
        assert checkFormat(
            in_mask_sh_points, dtype, device, [in_mask_sh_values.shape[0], 3], True
        )

        mask_boundary_sh_values = mash_cpp.toSHValues(
            sh_degree_max,
            sh_params,
            mask_boundary_phis,
            mask_boundary_thetas,
            mask_boundary_phi_idxs,
        )
        assert checkFormat(
            mask_boundary_sh_values,
            dtype,
            device,
            [mask_boundary_phis.shape[0]],
            True,
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
        assert checkFormat(
            mask_boundary_sh_points,
            dtype,
            device,
            [mask_boundary_sh_values.shape[0], 3],
            True,
        )
        now = timer.now()
        print("toSHValues+toSHPoints:", now)
        timer.reset()

        sample_point_num = ceil(in_mask_sh_points.shape[0] * sample_point_scale)

        v_in_mask_sh_points = in_mask_sh_points.reshape(1, -1, 3)

        float_detach_v_in_mask_sh_points = v_in_mask_sh_points.detach().type(
            torch.float32
        )

        v_fps_in_mask_sh_point_idxs = pointnet2_ops.furthest_point_sampling(
            float_detach_v_in_mask_sh_points, sample_point_num
        )

        fps_in_mask_sh_point_idxs = v_fps_in_mask_sh_point_idxs.reshape(-1)

        fps_in_mask_sh_points = in_mask_sh_points[fps_in_mask_sh_point_idxs]
        now = timer.now()
        print("toFPSPoints:", now)
        timer.reset()

        sh_points = torch.vstack([fps_in_mask_sh_points, mask_boundary_sh_points])
        print(
            "mash_unit:",
            in_mask_sh_points.shape,
            fps_in_mask_sh_points.shape,
            mask_boundary_sh_points.shape,
        )
        now = timer.now()
        print("vstack sh_points:", now)
        timer.reset()

    return True
