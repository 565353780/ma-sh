#pragma once

#include <torch/extension.h>

const torch::Tensor toBoundIdxs(const torch::Tensor &data_counts);
