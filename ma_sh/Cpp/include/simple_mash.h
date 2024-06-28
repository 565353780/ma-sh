#pragma once

#include <torch/extension.h>

const std::vector<torch::Tensor> toSimpleMashSamplePoints(
    const int &anchor_num, const int &mask_degree_max, const int &sh_degree_max,
    const torch::Tensor &mask_params, const torch::Tensor &sh_params,
    const torch::Tensor &rotate_vectors, const torch::Tensor &positions,
    const int &sample_phi_num, const int &sample_theta_num,
    const bool &use_inv = true);
