#pragma once

#include <torch/extension.h>

const torch::Tensor toMashSamplePoints(
    const int &anchor_num, const int &sh_degree_max,
    const torch::Tensor &mask_params, const torch::Tensor &sh_params,
    const torch::Tensor &rotate_vectors, const torch::Tensor &positions,
    const torch::Tensor &mask_boundary_phis,
    const torch::Tensor &mask_boundary_base_values,
    const torch::Tensor &mask_boundary_phi_idxs,
    const int &inner_sample_row_num, const float &sample_point_scale,
    const bool &use_inv = true);
