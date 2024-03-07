#pragma once

#include <torch/extension.h>

const torch::Tensor
toMaskBoundaryMaxThetas(const torch::Tensor &mask_boundary_thetas,
                        const torch::Tensor &mask_boundary_phi_idxs);
