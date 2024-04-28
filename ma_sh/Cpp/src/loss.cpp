#include "loss.h"
#include "chamfer.h"
#include "constant.h"

const torch::Tensor
toAnchorFitLoss(const int &anchor_num, const int &mask_boundary_sample_point_num,
          const torch::Tensor &fit_dists2,
          const torch::Tensor &mask_boundary_sample_phi_idxs,
          const torch::Tensor &in_mask_sample_point_idxs) {
  const torch::Tensor fit_dists = torch::sqrt(fit_dists2 + EPSILON);

  const torch::TensorOptions opts = torch::TensorOptions()
                                        .dtype(fit_dists2.dtype())
                                        .device(fit_dists2.device());

  torch::Tensor fit_loss = torch::zeros({anchor_num}, opts);

  for (int i = 0; i < anchor_num; ++i) {
    const torch::Tensor current_boundary_point_idxs =
        torch::where(mask_boundary_sample_phi_idxs == i)[0];

    const torch::Tensor current_inner_point_idxs =
        torch::where(in_mask_sample_point_idxs == i)[0] +
        mask_boundary_sample_point_num;

    const torch::Tensor current_push_idxs =
        torch::hstack({current_boundary_point_idxs, current_inner_point_idxs});

    const torch::Tensor current_fit_dists =
        fit_dists.index({current_push_idxs});

    const torch::Tensor current_mean_fit_dist = torch::mean(current_fit_dists);

    fit_loss.index_put_({i}, current_mean_fit_dist);
  }

  return fit_loss;
}

const torch::Tensor
toAnchorCoverageLoss(const int &anchor_num, const int &mask_boundary_sample_point_num,
               const torch::Tensor &coverage_dists2,
               const torch::Tensor &coverage_idxs,
               const torch::Tensor &mask_boundary_sample_phi_idxs,
               const torch::Tensor &in_mask_sample_point_idxs) {
  const torch::Tensor coverage_dists = torch::sqrt(coverage_dists2 + EPSILON);

  const torch::Tensor pull_boundary_mask =
      coverage_idxs < mask_boundary_sample_point_num;

  const torch::Tensor pull_boundary_idxs = torch::where(pull_boundary_mask)[0];

  const torch::Tensor pull_inner_idxs =
      torch::where(~pull_boundary_mask)[0] - mask_boundary_sample_point_num;

  const torch::Tensor pull_boundary_point_idxs = coverage_idxs.index({pull_boundary_idxs});

  const torch::Tensor pull_inner_point_idxs = coverage_idxs.index({pull_inner_idxs}) - mask_boundary_sample_point_num;

  const torch::Tensor pull_boundary_anchor_idxs =
      mask_boundary_sample_phi_idxs.index({pull_boundary_point_idxs});

  const torch::Tensor pull_inner_anchor_idxs =
      in_mask_sample_point_idxs.index({pull_inner_point_idxs});

  const torch::TensorOptions opts = torch::TensorOptions()
                                        .dtype(coverage_dists2.dtype())
                                        .device(coverage_dists2.device());

  torch::Tensor coverage_loss = torch::zeros({anchor_num}, opts);

  for (int i = 0; i < anchor_num; ++i) {
    const torch::Tensor current_pull_boundary_idxs =
        torch::where(pull_boundary_anchor_idxs == i)[0];

    const torch::Tensor current_pull_inner_idxs =
        torch::where(pull_inner_anchor_idxs == i)[0] + mask_boundary_sample_point_num;

    const torch::Tensor current_pull_idxs =
        torch::hstack({current_pull_boundary_idxs, current_pull_inner_idxs});

    if (current_pull_idxs.size(0) == 0) {
      continue;
    }

    const torch::Tensor current_coverage_dists =
        coverage_dists.index({current_pull_idxs});

    const torch::Tensor current_mean_coverage_dist =
        torch::mean(current_coverage_dists);

    coverage_loss.index_put_({i}, current_mean_coverage_dist);
  }

  return coverage_loss;
}

const std::vector<torch::Tensor>
toChamferDistanceLoss(const torch::Tensor &detect_points,
                      const torch::Tensor &gt_points) {
  const torch::Tensor v_detect_points =
      detect_points.unsqueeze(0).toType(gt_points.scalar_type());

  const std::vector<torch::Tensor> chamfer_distances =
      toChamferDistance(v_detect_points, gt_points);

  const torch::Tensor fit_dists2 = chamfer_distances[0].squeeze(0);
  const torch::Tensor coverage_dists2 = chamfer_distances[1].squeeze(0);

  const torch::Tensor fit_dists = torch::sqrt(fit_dists2 + EPSILON);
  const torch::Tensor coverage_dists = torch::sqrt(coverage_dists2 + EPSILON);

  const torch::Tensor fit_loss = torch::mean(fit_dists);
  const torch::Tensor coverage_loss = torch::mean(coverage_dists);

  const std::vector<torch::Tensor> chamfer_distance_losses(
      {fit_loss, coverage_loss});

  return chamfer_distance_losses;
}

const std::vector<torch::Tensor>
toAnchorChamferDistanceLoss(const int &anchor_num,
                      const torch::Tensor &mask_boundary_sample_points,
                      const torch::Tensor &in_mask_sample_points,
                      const torch::Tensor &mask_boundary_sample_phi_idxs,
                      const torch::Tensor &in_mask_sample_point_idxs,
                      const torch::Tensor &gt_points) {
  const torch::Tensor sample_points =
      torch::vstack({mask_boundary_sample_points, in_mask_sample_points});

  const torch::Tensor v_sample_points =
      sample_points.unsqueeze(0).toType(gt_points.scalar_type());

  const std::vector<torch::Tensor> chamfer_distances =
      toChamferDistance(v_sample_points, gt_points);

  const torch::Tensor fit_dists2 = chamfer_distances[0].squeeze(0);
  const torch::Tensor coverage_dists2 = chamfer_distances[1].squeeze(0);
  const torch::Tensor coverage_idxs = chamfer_distances[3].squeeze(0);

  const int mask_boundary_sample_point_num =
      mask_boundary_sample_points.size(0);

  const torch::Tensor anchor_fit_loss =
      toAnchorFitLoss(anchor_num, mask_boundary_sample_point_num, fit_dists2,
                mask_boundary_sample_phi_idxs, in_mask_sample_point_idxs);

  const torch::Tensor anchor_coverage_loss = toAnchorCoverageLoss(
      anchor_num, mask_boundary_sample_point_num, coverage_dists2,
      coverage_idxs, mask_boundary_sample_phi_idxs, in_mask_sample_point_idxs);

  const std::vector<torch::Tensor> chamfer_distance_losses(
      {anchor_fit_loss, anchor_coverage_loss});

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

  torch::Tensor boundary_connect_loss = torch::tensor(0.0, opts);

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
