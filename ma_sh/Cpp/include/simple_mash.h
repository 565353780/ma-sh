#pragma once

#include <torch/extension.h>

const torch::Tensor toSimpleMashSamplePoints(
    const int &anchor_num, const int &mask_degree_max, const int &sh_degree_max,
    const torch::Tensor &mask_params, const torch::Tensor &sh_params,
    const torch::Tensor &rotate_vectors, const torch::Tensor &positions,
    const torch::Tensor &sample_phis, const torch::Tensor &sample_base_values,
    const int &sample_theta_num, const bool &use_inv = true);
