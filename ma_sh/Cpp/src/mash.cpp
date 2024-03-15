#include "mash.h"
#include "mash_unit.h"

const torch::Tensor toMashSamplePoints(
    const int &sh_degree_max, const torch::Tensor &mask_params,
    const torch::Tensor &sh_params, const torch::Tensor &rotate_vectors,
    const torch::Tensor &positions, const torch::Tensor &sample_phis,
    const torch::Tensor &sample_thetas, const torch::Tensor &mask_boundary_phis,
    const torch::Tensor &mask_boundary_phi_idxs,
    const torch::Tensor &mask_boundary_phi_data_idxs,
    const torch::Tensor &mask_boundary_base_values,
    const torch::Tensor &sample_base_values,
    const torch::Tensor &sample_sh_directions) {
  torch::Tensor mask_boundary_thetas;
  toMaskBoundaryThetas(mask_params, mask_boundary_base_values,
                       mask_boundary_phi_idxs, mask_boundary_thetas);

  torch::Tensor in_max_mask_sample_polar_idxs;
  torch::Tensor in_max_mask_sample_polar_data_idxs;
  toInMaxMaskIdxs(sample_thetas, mask_boundary_thetas, mask_boundary_phi_idxs,
                  in_max_mask_sample_polar_idxs,
                  in_max_mask_sample_polar_data_idxs);

  torch::Tensor in_max_mask_sample_phis;
  torch::Tensor in_max_mask_sample_thetas;
  toInMaxMaskPolars(sample_phis, sample_thetas,
                    in_max_mask_sample_polar_data_idxs, in_max_mask_sample_phis,
                    in_max_mask_sample_thetas);

  torch::Tensor in_max_mask_base_values;
  toInMaxMaskBaseValues(sample_base_values, in_max_mask_sample_polar_data_idxs,
                        in_max_mask_base_values);

  torch::Tensor in_max_mask_thetas;
  toInMaxMaskThetas(mask_params, in_max_mask_base_values,
                    in_max_mask_sample_polar_idxs,
                    in_max_mask_sample_polar_data_idxs, in_max_mask_thetas);

  torch::Tensor in_mask_sample_phis;
  torch::Tensor in_mask_sample_polar_idxs;
  torch::Tensor in_mask_sample_polar_data_idxs;
  torch::Tensor in_mask_base_values;
  torch::Tensor in_mask_sample_theta_weights;

  toInMaskSamplePolarWeights(
      mask_params, in_max_mask_base_values, in_max_mask_thetas,
      in_max_mask_sample_phis, in_max_mask_sample_thetas,
      in_max_mask_sample_polar_idxs, in_max_mask_sample_polar_data_idxs,
      in_mask_sample_phis, in_mask_sample_polar_idxs,
      in_mask_sample_polar_data_idxs, in_mask_base_values,
      in_mask_sample_theta_weights);

  torch::Tensor detect_thetas;
  toSamplePolars(mask_params, in_mask_base_values, in_mask_sample_polar_idxs,
                 in_mask_sample_polar_data_idxs, detect_thetas);

  const torch::Tensor all_sample_phis =
      torch::hstack({in_mask_sample_phis, mask_boundary_phis});
  const torch::Tensor all_sample_thetas =
      torch::hstack({detect_thetas, mask_boundary_thetas});
  const torch::Tensor all_sample_polar_idxs =
      torch::hstack({in_mask_sample_polar_idxs, mask_boundary_phi_idxs});
  const torch::Tensor all_sample_polar_data_idxs = torch::hstack(
      {in_mask_sample_polar_data_idxs, mask_boundary_phi_data_idxs});

  torch::Tensor sh_values;
  toSHValues(sh_degree_max, sh_params, all_sample_phis, all_sample_thetas,
             all_sample_polar_idxs, sh_values);

  torch::Tensor sh_points;
  toSHPoints(rotate_vectors, positions, sample_sh_directions, sh_values,
             all_sample_polar_idxs, all_sample_polar_data_idxs, sh_points);

  return sh_points;
}
