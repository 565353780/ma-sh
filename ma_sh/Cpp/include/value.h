#pragma once

#include <torch/extension.h>

const torch::Tensor toValues(const torch::Tensor &params,
                             const torch::Tensor &base_values,
                             const torch::Tensor &phi_idxs);
