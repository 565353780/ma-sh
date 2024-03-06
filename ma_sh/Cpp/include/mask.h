#pragma once

#include <torch/extension.h>

const torch::Tensor getMaskBaseValues(const int &degree_max,
                                      const torch::Tensor &phis);
