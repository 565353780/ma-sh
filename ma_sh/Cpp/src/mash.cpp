#include "mash.h"
#include "constant.h"
#include "mash_unit.h"

const std::vector<torch::Tensor> toMashSamplePoints(
    const int &anchor_num, const int &sh_degree_max,
    const torch::Tensor &mask_params, const torch::Tensor &sh_params,
    const torch::Tensor &rotate_vectors, const torch::Tensor &positions,
    const torch::Tensor &sample_phis, const torch::Tensor &sample_thetas,
    const torch::Tensor &mask_boundary_phis,
    const torch::Tensor &mask_boundary_phi_idxs,
    const torch::Tensor &mask_boundary_phi_data_idxs,
    const torch::Tensor &mask_boundary_base_values,
    const torch::Tensor &sample_base_values,
    const torch::Tensor &sample_sh_directions,
    const float &sample_point_scale) {
  const torch::Tensor mask_boundary_thetas = toMaskBoundaryThetas(
      mask_params, mask_boundary_base_values, mask_boundary_phi_idxs);

  const std::vector<torch::Tensor> in_max_mask_sample_polar_idxs_vec =
      toInMaxMaskSamplePolarIdxsVec(anchor_num, sample_thetas,
                                    mask_boundary_thetas,
                                    mask_boundary_phi_idxs);

  const torch::Tensor in_max_mask_sample_polar_idxs =
      toInMaxMaskSamplePolarIdxs(in_max_mask_sample_polar_idxs_vec);

  const torch::Tensor in_max_mask_sample_polar_data_idxs =
      torch::hstack(in_max_mask_sample_polar_idxs_vec);

  const torch::Tensor in_max_mask_sample_phis =
      sample_phis.index({in_max_mask_sample_polar_data_idxs});

  const torch::Tensor in_max_mask_sample_thetas =
      sample_thetas.index({in_max_mask_sample_polar_data_idxs});

  const torch::Tensor in_max_mask_base_values =
      sample_base_values.index({slice_all, in_max_mask_sample_polar_data_idxs});

  const torch::Tensor in_max_mask_thetas = toInMaxMaskThetas(
      mask_params, in_max_mask_base_values, in_max_mask_sample_polar_idxs,
      in_max_mask_sample_polar_data_idxs);

  const torch::Tensor in_mask_sample_polar_mask =
      in_max_mask_sample_thetas <= in_max_mask_thetas;

  const torch::Tensor in_mask_sample_phis =
      in_max_mask_sample_phis.index({in_mask_sample_polar_mask});

  const torch::Tensor in_mask_sample_polar_idxs =
      in_max_mask_sample_polar_idxs.index({in_mask_sample_polar_mask});

  const torch::Tensor in_mask_sample_polar_data_idxs =
      in_max_mask_sample_polar_data_idxs.index({in_mask_sample_polar_mask});

  const torch::Tensor in_mask_base_values =
      in_max_mask_base_values.index({slice_all, in_mask_sample_polar_mask});

  const torch::Tensor in_mask_sample_theta_weights = toInMaskSampleThetaWeights(
      in_max_mask_sample_thetas, in_max_mask_thetas, in_mask_sample_polar_mask);

  const torch::Tensor detect_thetas =
      toDetectThetas(mask_params, in_mask_base_values,
                     in_mask_sample_polar_idxs, in_mask_sample_theta_weights);

  const torch::Tensor in_mask_sh_values =
      toSHValues(sh_degree_max, sh_params, in_mask_sample_phis, detect_thetas,
                 in_mask_sample_polar_idxs);

  const torch::Tensor in_mask_sh_points =
      toSHPoints(sh_params, rotate_vectors, positions, sample_sh_directions,
                 in_mask_sh_values, in_mask_sample_polar_idxs,
                 in_mask_sample_polar_data_idxs);

  const torch::Tensor mask_boundary_sh_values =
      toSHValues(sh_degree_max, sh_params, mask_boundary_phis,
                 mask_boundary_thetas, mask_boundary_phi_idxs);

  const torch::Tensor mask_boundary_sh_points =
      toSHPoints(sh_params, rotate_vectors, positions, sample_sh_directions,
                 mask_boundary_sh_values, mask_boundary_phi_idxs,
                 mask_boundary_phi_data_idxs);

  return std::vector<torch::Tensor>{in_mask_sh_points, mask_boundary_sh_points};
}
