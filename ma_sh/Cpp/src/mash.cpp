#include "mash.h"
#include "constant.h"
#include "direction.h"
#include "fps.h"
#include "mash_unit.h"
#include "mask.h"
#include "timer.h"

const std::vector<torch::Tensor>
toInMaskSamplePolars(const int &anchor_num, const torch::Tensor &mask_params,
                     const torch::Tensor &sample_phis,
                     const torch::Tensor &sample_thetas,
                     const torch::Tensor &mask_boundary_thetas,
                     const torch::Tensor &mask_boundary_phi_idxs,
                     const torch::Tensor &sample_base_values) {
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

  const torch::Tensor in_max_mask_sample_base_values =
      sample_base_values.index({slice_all, in_max_mask_sample_polar_data_idxs});

  const torch::Tensor in_max_mask_thetas = toInMaxMaskThetas(
      mask_params, in_max_mask_sample_base_values,
      in_max_mask_sample_polar_idxs, in_max_mask_sample_polar_data_idxs);

  const torch::Tensor in_mask_sample_polar_mask =
      in_max_mask_sample_thetas <= in_max_mask_thetas;

  const torch::Tensor in_mask_sample_phis =
      in_max_mask_sample_phis.index({in_mask_sample_polar_mask});

  const torch::Tensor in_mask_sample_polar_idxs =
      in_max_mask_sample_polar_idxs.index({in_mask_sample_polar_mask});

  const torch::Tensor in_mask_sample_polar_data_idxs =
      in_max_mask_sample_polar_data_idxs.index({in_mask_sample_polar_mask});

  const torch::Tensor in_mask_sample_base_values =
      in_max_mask_sample_base_values.index(
          {slice_all, in_mask_sample_polar_mask});

  const torch::Tensor in_mask_sample_theta_weights = toInMaskSampleThetaWeights(
      in_max_mask_sample_thetas, in_max_mask_thetas, in_mask_sample_polar_mask);

  const std::vector<torch::Tensor> in_mask_sample_polars_with_idxs(
      {in_mask_sample_phis, in_mask_sample_theta_weights,
       in_mask_sample_polar_idxs, in_mask_sample_base_values});
  return in_mask_sample_polars_with_idxs;
}

const torch::Tensor
toSamplePoints(const int &mask_degree_max, const int &sh_degree_max,
               const torch::Tensor &sh_params,
               const torch::Tensor &rotate_vectors,
               const torch::Tensor &positions, const torch::Tensor &sample_phis,
               const torch::Tensor &sample_thetas,
               const torch::Tensor &sample_polar_idxs, const bool &use_inv) {
  const torch::Tensor sh_values = toSHValues(
      sh_degree_max, sh_params, sample_phis, sample_thetas, sample_polar_idxs);

  const torch::Tensor sample_sh_directions =
      toDirections(sample_phis, sample_thetas);

  const torch::Tensor sh_points =
      toSHPoints(sh_params, rotate_vectors, positions, sample_sh_directions,
                 sh_values, sample_polar_idxs, use_inv);

  return sh_points;
}

const torch::Tensor toWeightedSamplePoints(
    const int &mask_degree_max, const int &sh_degree_max,
    const torch::Tensor &mask_params, const torch::Tensor &sh_params,
    const torch::Tensor &rotate_vectors, const torch::Tensor &positions,
    const torch::Tensor &sample_phis, const torch::Tensor &sample_theta_weights,
    const torch::Tensor &sample_polar_idxs, const bool &use_inv,
    const torch::Tensor &sample_base_values) {
  torch::Tensor valid_sample_base_values = sample_base_values;
  if (sample_base_values.size(0) != sample_phis.size(0)) {
    valid_sample_base_values = toMaskBaseValues(sample_phis, mask_degree_max);
  }

  const torch::Tensor detect_thetas =
      toDetectThetas(mask_params, valid_sample_base_values, sample_polar_idxs,
                     sample_theta_weights);

  const torch::Tensor sh_points = toSamplePoints(
      mask_degree_max, sh_degree_max, sh_params, rotate_vectors, positions,
      sample_phis, detect_thetas, sample_polar_idxs, use_inv);

  return sh_points;
}

const std::vector<torch::Tensor> toMashSamplePoints(
    const int &anchor_num, const int &mask_degree_max, const int &sh_degree_max,
    const torch::Tensor &mask_params, const torch::Tensor &sh_params,
    const torch::Tensor &rotate_vectors, const torch::Tensor &positions,
    const torch::Tensor &sample_phis, const torch::Tensor &sample_thetas,
    const torch::Tensor &mask_boundary_phis,
    const torch::Tensor &mask_boundary_phi_idxs,
    const torch::Tensor &mask_boundary_base_values,
    const torch::Tensor &sample_base_values, const float &sample_point_scale,
    const bool &use_inv) {
  Timer timer;
  const torch::Tensor mask_boundary_thetas = toMaskBoundaryThetas(
      mask_params, mask_boundary_base_values, mask_boundary_phi_idxs);

  timer.reset();
  const std::vector<torch::Tensor> in_mask_sample_polars_with_idxs =
      toInMaskSamplePolars(anchor_num, mask_params, sample_phis, sample_thetas,
                           mask_boundary_thetas, mask_boundary_phi_idxs,
                           sample_base_values);
  std::cout << "toInMaskSamplePolars : " << timer.now() << std::endl;

  const torch::Tensor &in_mask_sample_phis = in_mask_sample_polars_with_idxs[0];
  const torch::Tensor &in_mask_sample_theta_weights =
      in_mask_sample_polars_with_idxs[1];
  const torch::Tensor &in_mask_sample_polar_idxs =
      in_mask_sample_polars_with_idxs[2];
  const torch::Tensor &in_mask_sample_base_values =
      in_mask_sample_polars_with_idxs[3];

  timer.reset();
  const torch::Tensor in_mask_sh_points = toWeightedSamplePoints(
      mask_degree_max, sh_degree_max, mask_params, sh_params, rotate_vectors,
      positions, in_mask_sample_phis, in_mask_sample_theta_weights,
      in_mask_sample_polar_idxs, use_inv, in_mask_sample_base_values);
  std::cout << "toWeightedSamplePoints : " << timer.now() << std::endl;

  timer.reset();
  const torch::Tensor fps_in_mask_sample_point_idxs =
      toFPSPointIdxs(in_mask_sh_points, in_mask_sample_polar_idxs,
                     sample_point_scale, anchor_num);
  std::cout << "toFPSPointIdxs : " << timer.now() << std::endl;

  const torch::Tensor in_mask_sample_points =
      in_mask_sh_points.index({fps_in_mask_sample_point_idxs});

  const torch::Tensor in_mask_sample_point_idxs =
      in_mask_sample_polar_idxs.index({fps_in_mask_sample_point_idxs});

  timer.reset();
  const torch::Tensor mask_boundary_sample_points =
      toSamplePoints(mask_degree_max, sh_degree_max, sh_params, rotate_vectors,
                     positions, mask_boundary_phis, mask_boundary_thetas,
                     mask_boundary_phi_idxs, use_inv);
  std::cout << "toSamplePoints : " << timer.now() << std::endl;

  const std::vector<torch::Tensor> sample_points_with_idxs(
      {mask_boundary_sample_points, in_mask_sample_points,
       in_mask_sample_point_idxs});

  return sample_points_with_idxs;
}
