#pragma once

#include <torch/extension.h>
#include <torch/types.h>

const torch::Tensor toCounts(const std::vector<torch::Tensor> &data_vec);

const torch::Tensor toIdxs(const torch::Tensor &data_counts);

const std::vector<torch::Tensor>
toLowerIdxsVec(const torch::Tensor &values, const torch::Tensor &max_bounds);
