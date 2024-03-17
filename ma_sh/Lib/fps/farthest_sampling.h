#pragma once

#include <torch/extension.h>

const torch::Tensor
samplePointCloudsCuda(const torch::Tensor &points,
                      const torch::Tensor &point_counts,
                      const torch::Tensor &sample_point_nums);

const torch::Tensor farthestPointSamplingLauncher(
    const unsigned &batch_size, const unsigned &sampling_point_num,
    const unsigned *point_nums, const float *input_point_clouds);
