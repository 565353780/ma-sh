#pragma once

#include <torch/extension.h>

const torch::Tensor toDirections(const torch::Tensor &phis,
                                 const torch::Tensor &thetas);

const torch::Tensor toPolars(const torch::Tensor &directions);
