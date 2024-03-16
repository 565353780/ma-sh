#pragma once

#include <torch/extension.h>

const torch::Tensor furthest_point_sampling(const torch::Tensor &points,
                                            const int &nsamples);
