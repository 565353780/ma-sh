#pragma once

#include <c10/core/DeviceType.h>
#include <torch/extension.h>
#include <torch/types.h>

const torch::Tensor getUniformSamplePhis(const int &point_num);

const torch::Tensor getUniformSampleThetas(const torch::Tensor &phis);
