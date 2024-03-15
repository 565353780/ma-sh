#pragma once

#include <torch/extension.h>
#include <torch/types.h>

const torch::Tensor toCounts(const std::vector<torch::Tensor> &data_vec);

const torch::Tensor toIdxs(const torch::Tensor &data_counts);

const torch::Tensor toDataIdxs(const int &repeat_num, const int &idx_num);

const std::vector<torch::Tensor>
toLowerIdxsVec(const torch::Tensor &values, const torch::Tensor &max_bounds);
