#pragma once

#include <torch/extension.h>

const torch::Tensor toAnchorBounds(const int &anchor_num, const torch::Tensor &mask_boundary_sample_points,
    const torch::Tensor &in_mask_sample_points,
    const torch::Tensor &mask_boundary_sample_point_idxs,
    const torch::Tensor &in_mask_sample_point_idxs);
