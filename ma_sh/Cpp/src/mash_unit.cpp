#include "mash_unit.h"
#include "filter.h"
#include "idx.h"
#include "inv.h"
#include "rotate.h"
#include "sampling.h"
#include "sh.h"
#include "value.h"

const torch::Tensor
toMaskBoundaryThetas(const torch::Tensor &mask_params,
                     const torch::Tensor &mask_boundary_base_values,
                     const torch::Tensor &mask_boundary_phi_idxs) {
  const torch::Tensor mask_boundary_thetas =
      toValues(mask_params, mask_boundary_base_values, mask_boundary_phi_idxs);

  return mask_boundary_thetas;
}

const std::vector<torch::Tensor>
toInMaxMaskSamplePolarIdxsVec(const int &anchor_num,
                              const torch::Tensor &sample_thetas,
                              const torch::Tensor &mask_boundary_thetas,
                              const torch::Tensor &mask_boundary_phi_idxs) {
  const torch::Tensor detach_mask_boundary_thetas =
      mask_boundary_thetas.detach();

  const torch::Tensor mask_boundary_max_thetas = toMaxValues(
      anchor_num, detach_mask_boundary_thetas, mask_boundary_phi_idxs);

  const std::vector<torch::Tensor> in_max_mask_sample_polar_idxs_vec =
      toLowerIdxsVec(sample_thetas, mask_boundary_max_thetas);

  return in_max_mask_sample_polar_idxs_vec;
}

const torch::Tensor toInMaxMaskSamplePolarIdxs(
    const std::vector<torch::Tensor> &in_max_mask_sample_polar_idxs_vec) {
  const torch::Tensor in_max_mask_sample_polar_counts =
      toCounts(in_max_mask_sample_polar_idxs_vec);

  const torch::Tensor in_max_mask_sample_polar_idxs =
      toIdxs(in_max_mask_sample_polar_counts);

  return in_max_mask_sample_polar_idxs;
}

const torch::Tensor
toInMaxMaskThetas(const torch::Tensor &mask_params,
                  const torch::Tensor &in_max_mask_base_values,
                  const torch::Tensor &in_max_mask_sample_polar_idxs,
                  const torch::Tensor &in_max_mask_sample_polar_data_idxs) {
  const torch::Tensor detach_mask_params = mask_params.detach();

  const torch::Tensor in_max_mask_thetas =
      toValues(detach_mask_params, in_max_mask_base_values,
               in_max_mask_sample_polar_idxs);

  return in_max_mask_thetas;
}

const torch::Tensor
toInMaskSampleThetaWeights(const torch::Tensor &in_max_mask_sample_thetas,
                           const torch::Tensor &in_max_mask_thetas,
                           const torch::Tensor &in_mask_sample_polar_mask) {
  const torch::Tensor in_mask_thetas =
      in_max_mask_thetas.index({in_mask_sample_polar_mask});

  const torch::Tensor in_mask_sample_thetas =
      in_max_mask_sample_thetas.index({in_mask_sample_polar_mask});

  const torch::Tensor in_mask_sample_theta_weights =
      in_mask_sample_thetas / in_mask_thetas;

  return in_mask_sample_theta_weights;
}

const torch::Tensor
toDetectThetas(const torch::Tensor &mask_params,
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
                               const torch::Tensor &phis,
                               const torch::Tensor &thetas,
                               const torch::Tensor &polar_idxs) {
  const torch::Tensor sh_base_values =
      toSHBaseValues(phis, thetas, sh_degree_max);

  const torch::Tensor sh_values =
      toValues(sh_params, sh_base_values, polar_idxs);

  return sh_values;
}

const torch::Tensor toSHPoints(const torch::Tensor &sh_params,
                               const torch::Tensor &rotate_vectors,
                               const torch::Tensor &positions,
                               const torch::Tensor &sample_sh_directions,
                               const torch::Tensor &sh_values,
                               const torch::Tensor &polar_idxs,
                               const torch::Tensor &polar_data_idxs) {
  const torch::Tensor v_sh_values = sh_values.reshape({-1, 1});

  const torch::Tensor index_sh_directions =
      sample_sh_directions.index({polar_data_idxs});

  const torch::Tensor sh_local_points = v_sh_values * index_sh_directions;

  const torch::Tensor sh_local_inv_points =
      toInvPoints(sh_params, sh_local_points, polar_idxs);

  const torch::Tensor index_rotate_vectors = rotate_vectors.index({polar_idxs});

  const torch::Tensor index_rotate_matrixs =
      toRotateMatrixs(index_rotate_vectors);

  const torch::Tensor v_sh_local_inv_points =
      sh_local_inv_points.reshape({-1, 1, 3});

  const torch::Tensor v_sh_local_inv_rotate_points =
      torch::matmul(v_sh_local_inv_points, index_rotate_matrixs);

  const torch::Tensor sh_local_inv_rotate_points =
      v_sh_local_inv_rotate_points.reshape({-1, 3});

  const torch::Tensor index_positions = positions.index({polar_idxs});

  const torch::Tensor sh_points = index_positions + sh_local_inv_rotate_points;

  return sh_points;
}

const torch::Tensor toFPSPoints(const torch::Tensor &points,
                                const torch::Tensor &point_idxs,
                                const float &sample_point_scale,
                                const int &idx_num) {
  const torch::Tensor point_counts =
      toIdxCounts(point_idxs, idx_num).toType(torch::kInt32);

  const torch::Tensor sample_point_nums =
      torch::ceil(point_counts * sample_point_scale).toType(torch::kInt32);

  const torch::Tensor float_detach_points =
      points.detach().toType(torch::kFloat32);

  const torch::Tensor fps_point_idxs = furthest_point_sampling(
      float_detach_points, point_counts, sample_point_nums);

  const torch::Tensor fps_points = points.index({fps_point_idxs});

  return fps_points;
}
