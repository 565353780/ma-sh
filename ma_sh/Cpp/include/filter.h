#pragma once

#include <torch/extension.h>

const torch::Tensor toMaxValues(const torch::Tensor &data,
                                const torch::Tensor &data_idxs);
