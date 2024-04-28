#pragma once

#include <torch/extension.h>

const torch::Tensor
toAnchorFitLoss(const int &anchor_num, const int &mask_boundary_sample_point_num,
          const torch::Tensor &fit_dists2,
          const torch::Tensor &mask_boundary_sample_phi_idxs,
          const torch::Tensor &in_mask_sample_point_idxs);

const torch::Tensor
toAnchorCoverageLoss(const int &anchor_num, const int &mask_boundary_sample_point_num,
               const torch::Tensor &coverage_dists2,
               const torch::Tensor &coverage_idxs,
               const torch::Tensor &mask_boundary_sample_phi_idxs,
               const torch::Tensor &in_mask_sample_point_idxs);

const std::vector<torch::Tensor>
toChamferDistanceLoss(const torch::Tensor &detect_points,
                      const torch::Tensor &gt_points);

const std::vector<torch::Tensor>
toAnchorChamferDistanceLoss(const int &anchor_num,
                      const torch::Tensor &mask_boundary_sample_points,
                      const torch::Tensor &in_mask_sample_points,
                      const torch::Tensor &mask_boundary_sample_phi_idxs,
                      const torch::Tensor &in_mask_sample_point_idxs,
                      const torch::Tensor &gt_points);

const torch::Tensor
toBoundaryConnectLoss(const int &anchor_num,
                      const torch::Tensor &mask_boundary_sample_points,
                      const torch::Tensor &mask_boundary_sample_phi_idxs);
