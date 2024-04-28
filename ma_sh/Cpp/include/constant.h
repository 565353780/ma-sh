#pragma once

#include <cmath>
#include <torch/extension.h>

using namespace torch::indexing;

const float EPSILON = 1e-6;

const float PI_2 = M_PI * 2.0;

const float SPLIT_VALUE = 1.0 + std::sqrt(5.0);

const float PHI_WEIGHT = M_PI * SPLIT_VALUE;

const Slice slice_all = Slice(None);
