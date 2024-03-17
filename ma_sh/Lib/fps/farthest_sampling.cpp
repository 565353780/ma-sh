#include "farthest_sampling.h"
#include <cstdint>

const torch::Tensor
samplePointCloudsCuda(const torch::Tensor &points,
                      const torch::Tensor &point_counts,
                      const torch::Tensor &sample_point_nums) {
  const torch::TensorOptions float_opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(points.device());

  const torch::TensorOptions idx_opts =
      torch::TensorOptions().dtype(torch::kInt32).device(points.device());

  torch::Tensor point_bound_idxs =
      torch::zeros({point_counts.sizes()[0] + 1}, idx_opts);

  std::int32_t sample_point_num_sum = 0;

  for (int i = 0; i < point_counts.sizes()[0]; ++i) {
    point_bound_idxs[i + 1] = point_bound_idxs[i] + point_counts[i];

    sample_point_num_sum += sample_point_nums[i].item<std::int32_t>();
  }

  torch::Tensor output = torch::zeros({sample_point_num_sum}, idx_opts);

  torch::Tensor tmp = torch::full({points.sizes(0)}, 1e10, float_opts);

  farthestPointSamplingLauncher(point_counts.size(0), sampling_point_num,
                                input_point_nums, input_point_cloud, distances,
                                output_indices);

  return output;
}
