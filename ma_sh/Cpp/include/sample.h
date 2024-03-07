#pragma once

#include <torch/extension.h>

const torch::Tensor toUniformSamplePhis(const int &point_num);

const torch::Tensor toUniformSampleThetas(const torch::Tensor &phis);

const torch::Tensor toMaskBoundaryPhis(const int &anchor_num,
                                       const int &mask_boundary_sample_num);
