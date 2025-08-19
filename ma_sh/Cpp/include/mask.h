#pragma once

#include <torch/extension.h>

const torch::Tensor toMaskBaseValues(const torch::Tensor &phis,
                                     const int &degree_max);
