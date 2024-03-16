#pragma once

#include <torch/extension.h>

const std::vector<torch::Tensor> toMashSamplePoints(
    const int &sh_degree_max, const torch::Tensor &mask_params,
    const torch::Tensor &sh_params, const torch::Tensor &rotate_vectors,
    const torch::Tensor &positions, const torch::Tensor &sample_phis,
    const torch::Tensor &sample_thetas, const torch::Tensor &mask_boundary_phis,
    const torch::Tensor &mask_boundary_phi_idxs,
    const torch::Tensor &mask_boundary_phi_data_idxs,
    const torch::Tensor &mask_boundary_base_values,
    const torch::Tensor &sample_base_values,
    const torch::Tensor &sample_sh_directions, const float &sample_point_scale);
