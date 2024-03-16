#pragma once

#include <cstdint>
#include <torch/extension.h>

const torch::Tensor toMaxValues(const int &unique_idx_num,
                                const torch::Tensor &data,
                                const torch::Tensor &data_idxs);
