#include "inv.h"
#include "constant.h"
#include "weights.h"

const torch::Tensor toInvPoints(const torch::Tensor &sh_params,
                                const torch::Tensor &sh_points,
                                const torch::Tensor &polar_idxs) {
  const torch::TensorOptions opts = torch::TensorOptions()
                                        .dtype(sh_params.dtype())
                                        .device(sh_params.device());

  const torch::Tensor first_sh_params = sh_params.index({polar_idxs, 0});

  const torch::Tensor first_sh_base_values = W0[0] * first_sh_params;

  torch::Tensor inv_centers = torch::zeros({polar_idxs.sizes()[0], 3}, opts);

  inv_centers.index_put_({slice_all, 2}, -1.0 * first_sh_base_values);

  const torch::Tensor in_inv_points = sh_points - inv_centers;

  const torch::Tensor in_inv_point_norms = torch::norm(in_inv_points, 2, 1);

  const torch::Tensor v_in_inv_point_norms =
      in_inv_point_norms.reshape({-1, 1});

  const torch::Tensor in_inv_point_directions =
      in_inv_points / v_in_inv_point_norms;

  const torch::Tensor v_first_sh_base_values =
      first_sh_base_values.reshape({-1, 1});

  const torch::Tensor v_in_inv_point_lengths = 4.0 * v_first_sh_base_values *
                                               v_first_sh_base_values /
                                               v_in_inv_point_norms;

  const torch::Tensor inv_points =
      inv_centers + v_in_inv_point_lengths * in_inv_point_directions;

  return inv_points;
}
