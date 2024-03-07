#pragma once

#include <torch/extension.h>

const torch::Tensor toBoundIdxs(const torch::Tensor &data_counts);

const torch::Tensor toMaskBoundaryPhis(const int &anchor_num,
                                       const int &mask_boundary_sample_num);
