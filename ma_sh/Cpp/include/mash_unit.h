#pragma once

#include <torch/extension.h>

const torch::Tensor
toMaskBoundaryThetas(const torch::Tensor &mask_params,
                     const torch::Tensor &mask_boundary_base_values,
                     const torch::Tensor &mask_boundary_phi_idxs);

const std::vector<torch::Tensor>
toInMaxMaskIdxs(const torch::Tensor &sample_thetas,
                const torch::Tensor &mask_boundary_thetas,
                const torch::Tensor &mask_boundary_phi_idxs);

const std::vector<torch::Tensor>
toInMaxMaskPolars(const torch::Tensor &sample_phis,
                  const torch::Tensor &sample_thetas,
                  const torch::Tensor &in_max_mask_sample_polar_data_idxs);

const torch::Tensor
toInMaxMaskBaseValues(const torch::Tensor &sample_base_values,
                      const torch::Tensor &in_max_mask_sample_polar_data_idxs);

const torch::Tensor
toInMaxMaskThetas(const torch::Tensor &mask_params,
                  const torch::Tensor &in_max_mask_base_values,
                  const torch::Tensor &in_max_mask_sample_polar_idxs,
                  const torch::Tensor &in_max_mask_sample_polar_data_idxs);

const std::vector<torch::Tensor> toInMaskSamplePolarWeights(
    const torch::Tensor &mask_params,
    const torch::Tensor &in_max_mask_base_values,
    const torch::Tensor &in_max_mask_thetas,
    const torch::Tensor &in_max_mask_sample_phis,
    const torch::Tensor &in_max_mask_sample_thetas,
    const torch::Tensor &in_max_mask_sample_polar_idxs,
    const torch::Tensor &in_max_mask_sample_polar_data_idxs);

const torch::Tensor
toSamplePolars(const torch::Tensor &mask_params,
               const torch::Tensor &in_mask_base_values,
               const torch::Tensor &in_mask_sample_polar_idxs,
               const torch::Tensor &in_mask_sample_theta_weights);

const torch::Tensor toSHValues(const int &sh_degree_max,
                               const torch::Tensor &sh_params,
                               const torch::Tensor &in_mask_sample_phis,
                               const torch::Tensor &detect_thetas,
                               const torch::Tensor &in_mask_sample_polar_idxs);

const torch::Tensor
toSHPoints(const torch::Tensor &rotate_vectors, const torch::Tensor &positions,
           const torch::Tensor &sample_sh_directions,
           const torch::Tensor &sh_values,
           const torch::Tensor &in_mask_sample_polar_idxs,
           const torch::Tensor &in_mask_sample_polar_data_idxs);
