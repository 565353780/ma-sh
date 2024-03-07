#pragma once

#include <torch/extension.h>

const torch::Tensor toMaskBaseValues(const int &degree_max,
                                     const torch::Tensor &phis);

const torch::Tensor toMaskValues(const torch::Tensor &phi_idxs,
                                  const torch::Tensor &params,
                                  const torch::Tensor &base_values);
