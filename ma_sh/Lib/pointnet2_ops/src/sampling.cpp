#include "sampling.h"

void furthest_point_sampling_kernel_wrapper(int b, int n, int m,
                                            const float *dataset, float *temp,
                                            int *idxs);

const torch::Tensor
furthest_point_sampling(const torch::Tensor &points,
                        const torch::Tensor &point_counts,
                        const torch::Tensor &sample_point_nums) {
  const torch::TensorOptions float_opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(points.device());

  const torch::TensorOptions idx_opts =
      torch::TensorOptions().dtype(torch::kInt32).device(points.device());

  torch::Tensor point_start_idxs =
      torch::zeros({point_counts.sizes()[0]}, idx_opts);

  torch::Tensor sample_point_start_idxs =
      torch::zeros({point_counts.sizes()[0]}, idx_opts);

  std::int32_t max_point_num = 0;
  std::int32_t max_sample_point_num = 0;

  for (int i = 0; i < point_counts.size(0) - 1; ++i) {
    const std::int32_t current_point_num = point_counts[i].item<std::int32_t>();

    point_start_idxs[i + 1] = point_start_idxs[i] + current_point_num;

    max_point_num = std::max(max_point_num, current_point_num);

    const std::int32_t current_sample_point_num =
        sample_point_nums[i].item<std::int32_t>();

    sample_point_start_idxs[i + 1] =
        sample_point_start_idxs[i] + current_sample_point_num;

    max_sample_point_num =
        std::max(max_sample_point_num, current_sample_point_num);
  }

  torch::Tensor output = torch::zeros(
      {sample_point_start_idxs[-1].item<std::int32_t>()}, idx_opts);

  torch::Tensor tmp = torch::full({points.sizes()[0]}, 1e10, float_opts);

  furthest_point_sampling_kernel_wrapper(
      point_counts.size(0), max_point_num, max_sample_point_num,
      points.data_ptr<float>(), point_counts.data_ptr<int>(),
      point_start_idxs.data_ptr<int>(), sample_point_start_idxs.data_ptr<int>(),
      tmp.data_ptr<float>(), output.data_ptr<int>());

  return output;
}
