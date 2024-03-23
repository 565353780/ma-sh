import torch
from tqdm import trange
from torchviz import make_dot

import mash_cpp

from ma_sh.Method.Mash.mash_unit import (
    toParams,
    toPreLoadMaskBoundaryPhiIdxs,
    toPreLoadBaseValues,
)
from ma_sh.Method.render import renderPoints
from ma_sh.Module.timer import Timer
from ma_sh.Test.init_values import initValues


def test():
    anchor_num = 100
    mask_degree_max = 1
    sh_degree_max = 3
    mask_boundary_sample_num = 10
    inner_sample_row_num = 10
    sample_point_scale = 0.5
    use_inv = True
    idx_dtype = torch.int64
    dtype = torch.float64
    device = "cuda:0"

    mask_params, sh_params, rotate_vectors, positions = toParams(
        anchor_num, mask_degree_max, sh_degree_max, dtype, device
    )

    initValues(mask_params, sh_params, rotate_vectors, positions, 2)

    mask_boundary_phi_idxs = toPreLoadMaskBoundaryPhiIdxs(
        anchor_num, mask_boundary_sample_num, idx_dtype, device
    )
    mask_boundary_phis, mask_boundary_base_values = toPreLoadBaseValues(
        anchor_num, mask_boundary_sample_num, mask_degree_max, dtype, device
    )

    def saveGraph(data, graph_name):
        mean_data = torch.mean(data)

        g = make_dot(
            mean_data,
            params={
                "mask_params": mask_params,
                "sh_params": sh_params,
                "rotate_vectors": rotate_vectors,
                "positions": positions,
            },
        )

        g.render("./output/" + graph_name + ".gv", view=False)
        return

    timer = Timer()

    for i in trange(10):
        timer.reset()
        mask_boundary_thetas = mash_cpp.toMaskBoundaryThetas(
            mask_params, mask_boundary_base_values, mask_boundary_phi_idxs
        )
        print("mask_boundary_thetas:", timer.now())

        if i == 0:
            saveGraph(mask_boundary_thetas, "1-mask_boundary_thetas")

        timer.reset()
        sample_theta_nums = (
            torch.ones_like(mask_boundary_phi_idxs) * inner_sample_row_num
        )
        print("toSampleThetaNums:", timer.now())

        timer.reset()
        sample_theta_idxs_in_phi_idxs = mash_cpp.toIdxs(sample_theta_nums)
        print("toIdxs:", timer.now())

        timer.reset()
        sample_thetas = mash_cpp.toSampleThetas(mask_boundary_thetas, sample_theta_nums)
        print("toSampleThetas:", timer.now())

        if i == 0:
            saveGraph(sample_thetas, "2-sample_thetas")

        timer.reset()
        sample_theta_idxs = mask_boundary_phi_idxs[sample_theta_idxs_in_phi_idxs]

        repeat_sample_phis = mask_boundary_phis[sample_theta_idxs_in_phi_idxs]
        print("idx:", timer.now())

        timer.reset()
        sample_sh_directions = mash_cpp.toSHDirections(
            repeat_sample_phis, sample_thetas
        )
        print("toSHDirections:", timer.now())

        if i == 0:
            saveGraph(sample_sh_directions, "3-sample_sh_directions")

        timer.reset()
        sample_sh_values = mash_cpp.toSHValues(
            sh_degree_max,
            sh_params,
            repeat_sample_phis,
            sample_thetas,
            sample_theta_idxs,
        )
        print("toSHValues:", timer.now())

        if i == 0:
            saveGraph(sample_sh_values, "4-sample_sh_values")

        timer.reset()
        sample_sh_points = mash_cpp.toSHPoints(
            sh_params,
            rotate_vectors,
            positions,
            sample_sh_directions,
            sample_sh_values,
            sample_theta_idxs,
            use_inv,
        )
        print("toSHPoints:", timer.now())

        if i == 0:
            saveGraph(sample_sh_points, "5-sample_sh_points")

        timer.reset()
        fps_sample_sh_points = mash_cpp.toFPSPoints(
            sample_sh_points, sample_theta_idxs, sample_point_scale, anchor_num
        )
        print("toFPSPoints:", timer.now())

        if i == 0:
            saveGraph(fps_sample_sh_points, "6-sh_points")

        print("sh_points:", fps_sample_sh_points.shape)

        # renderPoints(fps_sample_sh_points.detach().clone().cpu().numpy())

        print("================================")

    return True
