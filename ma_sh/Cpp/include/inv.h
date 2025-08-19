#pragma once

#include <torch/extension.h>

const torch::Tensor toInvPoints(const torch::Tensor &sh_params,
                                const torch::Tensor &sh_points,
                                const torch::Tensor &polar_idxs);
