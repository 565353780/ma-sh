#include "mash_unit.h"
#include "constant.h"
#include "filter.h"
#include "idx.h"
#include "rotate.h"
#include "sh.h"
#include "value.h"

const torch::Tensor
toMaskBoundaryThetas(const torch::Tensor &mask_params,
                     const torch::Tensor &mask_boundary_base_values,
                     const torch::Tensor &mask_boundary_phi_idxs) {
  torch::NoGradGuard no_grad;

  const torch::Tensor mask_boundary_thetas =
      toValues(mask_params, mask_boundary_base_values, mask_boundary_phi_idxs);

  return mask_boundary_thetas;
}

const std::vector<torch::Tensor>
toInMaxMaskIdxs(const torch::Tensor &sample_thetas,
                const torch::Tensor &mask_boundary_thetas,
                const torch::Tensor &mask_boundary_phi_idxs) {
  const torch::Tensor mask_boundary_max_thetas =
      toMaxValues(mask_boundary_thetas, mask_boundary_phi_idxs);

  const std::vector<torch::Tensor> in_max_mask_sample_polar_idxs_vec =
      toLowerIdxsVec(sample_thetas, mask_boundary_max_thetas);

  const torch::Tensor in_max_mask_sample_polar_counts =
      toCounts(in_max_mask_sample_polar_idxs_vec);

  const torch::Tensor in_max_mask_sample_polar_idxs =
      toIdxs(in_max_mask_sample_polar_counts);

  const torch::Tensor in_max_mask_sample_polar_data_idxs =
      torch::hstack(in_max_mask_sample_polar_idxs_vec);

  const std::vector<torch::Tensor> in_max_mask_idxs_vec{
      in_max_mask_sample_polar_idxs, in_max_mask_sample_polar_data_idxs};

  return in_max_mask_idxs_vec;
}

const std::vector<torch::Tensor>
toInMaxMaskPolars(const torch::Tensor &sample_phis,
                  const torch::Tensor &sample_thetas,
                  const torch::Tensor &in_max_mask_sample_polar_data_idxs) {
  const torch::Tensor in_max_mask_sample_phis =
      sample_phis.index({in_max_mask_sample_polar_data_idxs});

  const torch::Tensor in_max_mask_sample_thetas =
      sample_thetas.index({in_max_mask_sample_polar_data_idxs});

  const std::vector<torch::Tensor> in_max_mask_polars_vec{
      in_max_mask_sample_phis, in_max_mask_sample_thetas};

  return in_max_mask_polars_vec;
}

const torch::Tensor
toInMaxMaskBaseValues(const torch::Tensor &sample_base_values,
                      const torch::Tensor &in_max_mask_sample_polar_data_idxs) {
  const torch::Tensor in_max_mask_base_values =
      sample_base_values.index({slice_all, in_max_mask_sample_polar_data_idxs});

  return in_max_mask_base_values;
}

const torch::Tensor
toInMaxMaskThetas(const torch::Tensor &mask_params,
                  const torch::Tensor &in_max_mask_base_values,
                  const torch::Tensor &in_max_mask_sample_polar_idxs,
                  const torch::Tensor &in_max_mask_sample_polar_data_idxs) {
  torch::NoGradGuard no_grad;

  const torch::Tensor in_max_mask_thetas = toValues(
      mask_params, in_max_mask_base_values, in_max_mask_sample_polar_idxs);

  return in_max_mask_thetas;
}

const std::vector<torch::Tensor> toInMaskSamplePolarWeights(
    const torch::Tensor &mask_params,
    const torch::Tensor &in_max_mask_base_values,
    const torch::Tensor &in_max_mask_thetas,
    const torch::Tensor &in_max_mask_sample_phis,
    const torch::Tensor &in_max_mask_sample_thetas,
    const torch::Tensor &in_max_mask_sample_polar_idxs,
    const torch::Tensor &in_max_mask_sample_polar_data_idxs) {
  const torch::Tensor in_mask_sample_polar_mask =
      in_max_mask_sample_thetas <= in_max_mask_thetas;

  const torch::Tensor in_mask_sample_phis =
      in_max_mask_sample_phis.index({in_mask_sample_polar_mask});

  const torch::Tensor in_mask_sample_thetas =
      in_max_mask_sample_thetas.index({in_mask_sample_polar_mask});

  const torch::Tensor in_mask_sample_polar_idxs =
      in_max_mask_sample_polar_idxs.index({in_mask_sample_polar_mask});

  const torch::Tensor in_mask_sample_polar_data_idxs =
      in_max_mask_sample_polar_data_idxs.index({in_mask_sample_polar_mask});

  const torch::Tensor in_mask_base_values =
      in_max_mask_base_values.index({slice_all, in_mask_sample_polar_mask});

  const torch::Tensor in_mask_thetas =
      in_max_mask_thetas.index({in_mask_sample_polar_mask});

  const torch::Tensor in_mask_sample_theta_weights =
      in_mask_sample_thetas / in_mask_thetas;

  const std::vector<torch::Tensor> in_mask_sample_polar_weights_vec{
      in_mask_sample_phis, in_mask_sample_polar_idxs,
      in_mask_sample_polar_data_idxs, in_mask_base_values,
      in_mask_sample_theta_weights};

  return in_mask_sample_polar_weights_vec;
}

const torch::Tensor
toSamplePolars(const torch::Tensor &mask_params,
               const torch::Tensor &in_mask_base_values,
               const torch::Tensor &in_mask_sample_polar_idxs,
               const torch::Tensor &in_mask_sample_theta_weights) {
  const torch::Tensor detect_boundary_thetas =
      toValues(mask_params, in_mask_base_values, in_mask_sample_polar_idxs);

  const torch::Tensor detect_thetas =
      in_mask_sample_theta_weights * detect_boundary_thetas;

  return detect_thetas;
}

const torch::Tensor toSHValues(const int &sh_degree_max,
                               const torch::Tensor &sh_params,
                               const torch::Tensor &in_mask_sample_phis,
                               const torch::Tensor &detect_thetas,
                               const torch::Tensor &in_mask_sample_polar_idxs) {
  const torch::Tensor sh_base_values =
      toSHBaseValues(in_mask_sample_phis, detect_thetas, sh_degree_max);

  const torch::Tensor sh_values =
      toValues(sh_params, sh_base_values, in_mask_sample_polar_idxs);

  return sh_values;
}

const torch::Tensor
toSHPoints(const torch::Tensor &rotate_vectors, const torch::Tensor &positions,
           const torch::Tensor &sample_sh_directions,
           const torch::Tensor &sh_values,
           const torch::Tensor &in_mask_sample_polar_idxs,
           const torch::Tensor &in_mask_sample_polar_data_idxs) {
  const torch::Tensor v_sh_values = sh_values.reshape({-1, 1});

  const torch::Tensor in_mask_sh_directions =
      sample_sh_directions.index({in_mask_sample_polar_data_idxs});

  const torch::Tensor sh_local_points = v_sh_values * in_mask_sh_directions;

  const torch::Tensor in_mask_rotate_vectors =
      rotate_vectors.index({in_mask_sample_polar_idxs});

  const torch::Tensor in_mask_rotate_matrixs =
      toRotateMatrixs(in_mask_rotate_vectors);

  const torch::Tensor v_sh_local_points = sh_local_points.reshape({-1, 1, 3});

  const torch::Tensor v_sh_local_rotate_points =
      torch::matmul(v_sh_local_points, in_mask_rotate_matrixs);

  const torch::Tensor sh_local_rotate_points =
      v_sh_local_rotate_points.reshape({-1, 3});

  const torch::Tensor in_mask_positions =
      positions.index({in_mask_sample_polar_idxs});

  const torch::Tensor sh_points = in_mask_positions + sh_local_rotate_points;

  return sh_points;
}
