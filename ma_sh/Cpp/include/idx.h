#pragma once

#include <torch/extension.h>

const torch::Tensor toBoundIdxs(const torch::Tensor &data_counts);

const std::vector<torch::Tensor>
toInMaskSamplePolarIdxs(const torch::Tensor &sample_thetas,
                        const torch::Tensor &mask_boundary_max_thetas);
