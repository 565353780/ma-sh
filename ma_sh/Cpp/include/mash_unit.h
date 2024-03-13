#pragma once

#include <torch/extension.h>

void toMaskBoundaryThetas(const torch::Tensor &mask_params,
                          const torch::Tensor &mask_boundary_base_values,
                          const torch::Tensor &mask_boundary_phi_idxs,
                          torch::Tensor &mask_boundary_thetas);

void toInMaxMaskIdxs(const torch::Tensor &sample_thetas,
                     const torch::Tensor &mask_boundary_thetas,
                     const torch::Tensor &mask_boundary_phi_idxs,
                     torch::Tensor &in_max_mask_sample_polar_idxs,
                     torch::Tensor &in_max_mask_sample_polar_data_idxs);

void toInMaxMaskPolars(const torch::Tensor &sample_phis,
                       const torch::Tensor &sample_thetas,
                       const torch::Tensor &in_max_mask_sample_polar_data_idxs,
                       torch::Tensor &in_max_mask_sample_phis,
                       torch::Tensor &in_max_mask_sample_thetas);

void toInMaxMaskThetas(const torch::Tensor &mask_params,
                       const torch::Tensor &sample_base_values,
                       const torch::Tensor &in_max_mask_sample_polar_idxs,
                       const torch::Tensor &in_max_mask_sample_polar_data_idxs,
                       torch::Tensor &in_max_mask_base_values,
                       torch::Tensor &in_max_mask_thetas);

void toInMaskSamplePolarWeights(
    const torch::Tensor &mask_params,
    const torch::Tensor &in_max_mask_base_values,
    const torch::Tensor &in_max_mask_thetas,
    const torch::Tensor &in_max_mask_sample_phis,
    const torch::Tensor &in_max_mask_sample_thetas,
    const torch::Tensor &in_max_mask_sample_polar_idxs,
    const torch::Tensor &in_max_mask_sample_polar_data_idxs,
    torch::Tensor &in_mask_sample_phis,
    torch::Tensor &in_mask_sample_polar_idxs,
    torch::Tensor &in_mask_sample_polar_data_idxs,
    torch::Tensor &in_mask_base_values,
    torch::Tensor &in_mask_sample_theta_weights);

void toSamplePolars(const torch::Tensor &mask_params,
                    const torch::Tensor &in_mask_base_values,
                    const torch::Tensor &in_mask_sample_polar_idxs,
                    const torch::Tensor &in_mask_sample_theta_weights,
                    torch::Tensor &detect_thetas);

void toSHValues(const int &sh_degree_max, const torch::Tensor &sh_params,
                const torch::Tensor &in_mask_sample_phis,
                const torch::Tensor &detect_thetas,
                const torch::Tensor &in_mask_sample_polar_idxs,
                torch::Tensor &sh_values);

void toSHPoints(const torch::Tensor &rotate_vectors,
                const torch::Tensor &positions,
                const torch::Tensor &sample_sh_directions,
                const torch::Tensor &sh_values,
                const torch::Tensor &in_mask_sample_polar_idxs,
                const torch::Tensor &in_mask_sample_polar_data_idxs,
                torch::Tensor &sh_points);
