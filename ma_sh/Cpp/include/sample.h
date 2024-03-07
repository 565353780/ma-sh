#pragma once

#include <torch/extension.h>
#include <torch/types.h>

const torch::Tensor
getUniformSamplePhis(const int &point_num,
                     const torch::Dtype &dtype = torch::kFloat32,
                     const torch::Device &device = torch::kCPU);

const torch::Tensor getUniformSampleThetas(const torch::Tensor &phis);
