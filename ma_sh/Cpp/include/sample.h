#pragma once

#include <torch/extension.h>

const torch::Tensor getUniformSamplePhis(const int &point_num);

const torch::Tensor getUniformSampleThetas(const torch::Tensor &phis);
