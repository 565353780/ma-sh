#pragma once

#include <torch/extension.h>

const torch::Tensor toMaskBaseValues(const torch::Tensor &phis,
                                     const int &degree_max);

const torch::Tensor toMaskValues(const torch::Tensor &params,
                                 const torch::Tensor &base_values,
                                 const torch::Tensor &phi_idxs);
