#pragma once

#include <torch/extension.h>

const torch::Tensor toSingleRotateMatrix(const torch::Tensor &rotate_vector);

const torch::Tensor toSingleMaskBoundaryThetas(const int &mask_degree_max,
                                               const torch::Tensor &mask_param,
                                               const torch::Tensor &mask_phis);
