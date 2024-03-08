#pragma once

#include <torch/extension.h>

const torch::Tensor toBoundIdxs(const torch::Tensor &data_counts);

const std::vector<torch::Tensor>
toInMaxMaskSamplePolarIdxsVec(const torch::Tensor &sample_thetas,
                              const torch::Tensor &mask_boundary_max_thetas);

const torch::Tensor toInMaskSamplePolarCounts(
    const std::vector<torch::Tensor> &in_mask_sample_polar_idxs_vec);
