#pragma once

#include <torch/extension.h>

const torch::Tensor getBaseValues(const int &degree_max,
                                  const torch::Tensor &data_counts);
