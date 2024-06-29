#pragma once

#include <torch/extension.h>

const std::vector<torch::Tensor>
toInMaskSamplePolars(const int &anchor_num, const torch::Tensor &mask_params,
                     const torch::Tensor &sample_phis,
                     const torch::Tensor &sample_thetas,
                     const torch::Tensor &mask_boundary_thetas,
                     const torch::Tensor &mask_boundary_phi_idxs,
                     const torch::Tensor &sample_base_values);

const torch::Tensor toSamplePoints(
    const int &mask_degree_max, const int &sh_degree_max,
    const torch::Tensor &sh_params, const torch::Tensor &rotate_vectors,
    const torch::Tensor &positions, const torch::Tensor &sample_phis,
    const torch::Tensor &sample_thetas, const torch::Tensor &sample_polar_idxs,
    const bool &use_inv = true);

const torch::Tensor toWeightedSamplePoints(
    const int &mask_degree_max, const int &sh_degree_max,
    const torch::Tensor &mask_params, const torch::Tensor &sh_params,
    const torch::Tensor &rotate_vectors, const torch::Tensor &positions,
    const torch::Tensor &sample_phis, const torch::Tensor &sample_theta_weights,
    const torch::Tensor &sample_polar_idxs, const bool &use_inv = true,
    const torch::Tensor &sample_base_values = torch::Tensor());

const std::vector<torch::Tensor> toMashSamplePoints(
    const int &anchor_num, const int &mask_degree_max, const int &sh_degree_max,
    const torch::Tensor &mask_params, const torch::Tensor &sh_params,
    const torch::Tensor &rotate_vectors, const torch::Tensor &positions,
    const torch::Tensor &sample_phis, const torch::Tensor &sample_thetas,
    const torch::Tensor &mask_boundary_phis,
    const torch::Tensor &mask_boundary_phi_idxs,
    const torch::Tensor &mask_boundary_base_values,
    const torch::Tensor &sample_base_values, const float &sample_point_scale,
    const bool &use_inv = true);
