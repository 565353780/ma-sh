#include "mash.h"
#include "idx.h"
#include "mash_unit.h"
#include "sample.h"
#include "sh.h"

const torch::Tensor toMashSamplePoints(
    const int &anchor_num, const int &sh_degree_max,
    const torch::Tensor &mask_params, const torch::Tensor &sh_params,
    const torch::Tensor &rotate_vectors, const torch::Tensor &positions,
    const torch::Tensor &mask_boundary_phis,
    const torch::Tensor &mask_boundary_base_values,
    const torch::Tensor &mask_boundary_phi_idxs, const float &delta_theta_angle,
    const float &sample_point_scale, const bool &use_inv) {
  const torch::Tensor mask_boundary_thetas = toMaskBoundaryThetas(
      mask_params, mask_boundary_base_values, mask_boundary_phi_idxs);

  const torch::Tensor sample_theta_nums =
      toSampleThetaNums(mask_boundary_thetas, delta_theta_angle);

  const torch::Tensor sample_theta_idxs_in_phi_idxs = toIdxs(sample_theta_nums);

  const torch::Tensor sample_thetas =
      toSampleThetas(mask_boundary_thetas, sample_theta_nums);

  const torch::Tensor sample_theta_idxs =
      mask_boundary_phi_idxs.index({sample_theta_idxs_in_phi_idxs});

  const torch::Tensor repeat_sample_phis =
      mask_boundary_phis.index({sample_theta_idxs_in_phi_idxs});

  const torch::Tensor sample_sh_directions =
      toSHDirections(repeat_sample_phis, sample_thetas);

  const torch::Tensor sample_sh_values =
      toSHValues(sh_degree_max, sh_params, repeat_sample_phis, sample_thetas,
                 sample_theta_idxs);

  const torch::Tensor sample_sh_points =
      toSHPoints(sh_params, rotate_vectors, positions, sample_sh_directions,
                 sample_sh_values, sample_theta_idxs, use_inv);

  const torch::Tensor fps_sample_sh_points = toFPSPoints(
      sample_sh_points, sample_theta_idxs, sample_point_scale, anchor_num);

  return fps_sample_sh_points;
}
