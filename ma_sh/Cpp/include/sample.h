#pragma once

#include <torch/extension.h>

const torch::Tensor toUniformSamplePhis(const int &sample_num);

const torch::Tensor toUniformSampleThetas(const int &sample_num);

const torch::Tensor toMaskBoundaryPhis(const int &anchor_num,
                                       const int &mask_boundary_sample_num);
