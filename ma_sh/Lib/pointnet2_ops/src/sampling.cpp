#include "sampling.h"
#include "utils.h"

const torch::Tensor
furthest_point_sampling(const torch::Tensor &points,
                        const torch::Tensor &point_counts,
                        const torch::Tensor &sample_point_nums) {
  CHECK_INPUT(points);

  const torch::TensorOptions float_opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(points.device());

  const torch::TensorOptions idx_opts =
      torch::TensorOptions().dtype(torch::kInt32).device(points.device());

  torch::Tensor point_start_idxs =
      torch::zeros({point_counts.size(0)}, idx_opts);

  torch::Tensor sample_point_bound_idxs =
      torch::zeros({point_counts.size(0) + 1}, idx_opts);

  std::int32_t max_point_num = 0;
  std::int32_t max_sample_point_num = 0;

  for (int i = 0; i < point_counts.size(0) - 1; ++i) {
    const std::int32_t current_point_num = point_counts[i].item<std::int32_t>();

    point_start_idxs[i + 1] = point_start_idxs[i] + current_point_num;

    max_point_num = std::max(max_point_num, current_point_num);

    const std::int32_t current_sample_point_num =
        sample_point_nums[i].item<std::int32_t>();

    sample_point_bound_idxs[i + 1] =
        sample_point_bound_idxs[i] + current_sample_point_num;

    max_sample_point_num =
        std::max(max_sample_point_num, current_sample_point_num);
  }
  sample_point_bound_idxs[-1] =
      sample_point_bound_idxs[-2] + sample_point_nums[-1];

  torch::Tensor output = torch::zeros(
      {sample_point_bound_idxs[-1].item<std::int32_t>()}, idx_opts);

  torch::Tensor tmp = torch::full({points.size(0)}, 1e10, float_opts);

  std::cout << "fps state:\n" << std::endl;
  std::cout << "points.shape:" << points.sizes() << std::endl;
  std::cout << "point_counts:\n" << point_counts << std::endl;
  std::cout << "sample_point_nums:\n" << sample_point_nums << std::endl;

  furthest_point_sampling_kernel_wrapper(
      point_counts.size(0), max_point_num, max_sample_point_num,
      points.data_ptr<float>(), point_counts.data_ptr<int>(),
      sample_point_nums.data_ptr<int>(), point_start_idxs.data_ptr<int>(),
      sample_point_bound_idxs.data_ptr<int>(), tmp.data_ptr<float>(),
      output.data_ptr<int>());

  torch::Tensor fps_idxs = torch::zeros_like(output);

  for (int i = 0; i < point_counts.size(0); ++i) {
    const torch::indexing::Slice current_slice(
        sample_point_bound_idxs[i].item<std::int32_t>(),
        sample_point_bound_idxs[i + 1].item<std::int32_t>());

    fps_idxs.index_put_({current_slice},
                        output.index({current_slice}) + point_start_idxs[i]);
  }

  return fps_idxs;
}
