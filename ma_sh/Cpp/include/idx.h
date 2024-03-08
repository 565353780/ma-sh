#pragma once

#include <torch/extension.h>

const torch::Tensor toBoundIdxs(const torch::Tensor &data_counts);

const std::vector<torch::Tensor>
toLowerValueIdxsVec(const torch::Tensor &values,
                    const torch::Tensor &max_bounds);

const torch::Tensor toInMaskSamplePolarCounts(
    const std::vector<torch::Tensor> &in_mask_sample_polar_idxs_vec);
