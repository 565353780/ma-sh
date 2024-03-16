#pragma once

#include <torch/extension.h>

const torch::Tensor
toMaskBoundaryThetas(const torch::Tensor &mask_params,
                     const torch::Tensor &mask_boundary_base_values,
                     const torch::Tensor &mask_boundary_phi_idxs);

const std::vector<torch::Tensor>
toInMaxMaskSamplePolarIdxsVec(const torch::Tensor &sample_thetas,
                              const torch::Tensor &mask_boundary_thetas,
                              const torch::Tensor &mask_boundary_phi_idxs);

const torch::Tensor toInMaxMaskSamplePolarIdxs(
    const std::vector<torch::Tensor> &in_max_mask_sample_polar_idxs_vec);

const torch::Tensor
toInMaxMaskThetas(const torch::Tensor &mask_params,
                  const torch::Tensor &in_max_mask_base_values,
                  const torch::Tensor &in_max_mask_sample_polar_idxs,
                  const torch::Tensor &in_max_mask_sample_polar_data_idxs);

const torch::Tensor
toInMaskSampleThetaWeights(const torch::Tensor &in_max_mask_sample_thetas,
                           const torch::Tensor &in_max_mask_thetas,
                           const torch::Tensor &in_mask_sample_polar_mask);

const torch::Tensor
toDetectThetas(const torch::Tensor &mask_params,
               const torch::Tensor &in_mask_base_values,
               const torch::Tensor &in_mask_sample_polar_idxs,
               const torch::Tensor &in_mask_sample_theta_weights);

const torch::Tensor toSHValues(const int &sh_degree_max,
                               const torch::Tensor &sh_params,
                               const torch::Tensor &phis,
                               const torch::Tensor &thetas,
                               const torch::Tensor &polar_idxs);

const torch::Tensor toSHPoints(const torch::Tensor &sh_params,
                               const torch::Tensor &rotate_vectors,
                               const torch::Tensor &positions,
                               const torch::Tensor &sample_sh_directions,
                               const torch::Tensor &sh_values,
                               const torch::Tensor &polar_idxs,
                               const torch::Tensor &polar_data_idxs);
