#pragma once

#include <torch/extension.h>

void furthest_point_sampling_kernel_wrapper(
    int point_batch_num, int max_point_num, int max_sample_point_num,
    const float *points, const int *point_counts, const int *point_start_idxs,
    const int *sample_point_start_idxs, float *tmp, int *idxs);

const torch::Tensor furthest_point_sampling(const torch::Tensor &points,
                                            const int &nsamples);
