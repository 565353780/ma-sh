#pragma once

#include <torch/extension.h>

const torch::Tensor toChamferDistanceLoss(const torch::Tensor &detect_points,
                                          const torch::Tensor &gt_points);

const torch::Tensor
toBoundaryConnectLoss(const int &anchor_num,
                      const torch::Tensor &mask_boundary_sample_points,
                      const torch::Tensor &mask_boundary_sample_phi_idxs);
