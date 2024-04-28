#include "loss.h"
#include "chamfer.h"
#include "constant.h"

const torch::Tensor toChamferDistanceLoss(const torch::Tensor &detect_points,
                                          const torch::Tensor &gt_points) {
  const std::vector<torch::Tensor> chamfer_distances =
      toChamferDistance(detect_points, gt_points);

  const torch::Tensor &fit_dists2 = chamfer_distances[0];
  const torch::Tensor &coverage_dists2 = chamfer_distances[1];

  const torch::Tensor fit_dists = torch::sqrt(fit_dists2 + EPSILON);
  const torch::Tensor coverage_dists = torch::sqrt(coverage_dists2 + EPSILON);

  const torch::Tensor fit_loss = torch::mean(fit_dists);
  const torch::Tensor coverage_loss = torch::mean(coverage_dists);

  const torch::Tensor chamfer_distance_losses =
      torch::hstack({fit_loss, coverage_loss});

  return chamfer_distance_losses;
}

const torch::Tensor
toBoundaryConnectLoss(const int &anchor_num,
                      const torch::Tensor &mask_boundary_sample_points,
                      const torch::Tensor &mask_boundary_sample_phi_idxs) {
  const torch::TensorOptions opts =
      torch::TensorOptions()
          .dtype(mask_boundary_sample_points.dtype())
          .device(mask_boundary_sample_points.device());

  torch::Tensor boundary_connect_loss = torch::zeros({1}, opts);

  for (int i = 0; i < anchor_num; ++i) {
    const torch::Tensor current_boundary_point_mask =
        mask_boundary_sample_phi_idxs == i;

    const torch::Tensor current_boundary_points =
        mask_boundary_sample_points.index({current_boundary_point_mask});

    const torch::Tensor other_boundary_points =
        mask_boundary_sample_points.index({~current_boundary_point_mask});

    const torch::Tensor v_current_boundary_points =
        current_boundary_points.unsqueeze(0);

    const torch::Tensor v_other_boundary_points =
        other_boundary_points.unsqueeze(0);

    const std::vector<torch::Tensor> current_boundary_chamfer_distances =
        toChamferDistance(v_current_boundary_points, v_other_boundary_points);

    const torch::Tensor &current_boundary_connect_dists2 =
        current_boundary_chamfer_distances[0];

    const torch::Tensor current_boundary_connect_dists =
        torch::sqrt(current_boundary_connect_dists2 + EPSILON);

    const torch::Tensor current_boundary_connect_loss =
        torch::mean(current_boundary_connect_dists);

    boundary_connect_loss =
        boundary_connect_loss + current_boundary_connect_loss;
  }

  boundary_connect_loss = boundary_connect_loss / anchor_num;

  return boundary_connect_loss;
}
