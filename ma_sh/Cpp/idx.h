#pragma once

#include <torch/extension.h>

torch::Tensor toBoundIdxs(const torch::Tensor &data_counts);
