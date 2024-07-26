#include "bound.h"

const torch::Tensor
toAnchorBounds(const int &anchor_num,
               const torch::Tensor &mask_boundary_sample_points,
               const torch::Tensor &in_mask_sample_points,
               const torch::Tensor &mask_boundary_sample_point_idxs,
               const torch::Tensor &in_mask_sample_point_idxs) {
  const torch::TensorOptions opts =
      torch::TensorOptions()
          .dtype(mask_boundary_sample_points.dtype())
          .device(mask_boundary_sample_points.device());

  torch::Tensor anchor_bounds = torch::zeros({anchor_num, 2, 3}, opts);

  for (int i = 0; i < anchor_num; ++i) {
    const torch::Tensor current_boundary_point_mask =
        mask_boundary_sample_point_idxs == i;

    const torch::Tensor current_inner_point_mask =
        in_mask_sample_point_idxs == i;

    const torch::Tensor current_boundary_points =
        mask_boundary_sample_points.index({current_boundary_point_mask});

    const torch::Tensor current_inner_points =
        in_mask_sample_points.index({current_inner_point_mask});

    const torch::Tensor current_anchor_points =
        torch::vstack({current_boundary_points, current_inner_points});

    const torch::Tensor current_min_bound =
        std::get<0>(torch::min(current_anchor_points, 0));

    const torch::Tensor current_max_bound =
        std::get<0>(torch::max(current_anchor_points, 0));

    anchor_bounds.index_put_({i, 0}, current_min_bound);

    anchor_bounds.index_put_({i, 1}, current_max_bound);
  }

  return anchor_bounds;
}
