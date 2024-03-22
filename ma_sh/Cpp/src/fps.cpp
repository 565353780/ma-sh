#include "constant.h"
#include "fps.h"
#include "idx.h"
#include "utils.h"

#ifdef USE_CUDA
const torch::Tensor
furthest_point_sampling(const torch::Tensor &points,
                        const torch::Tensor &point_counts,
                        const torch::Tensor &sample_point_nums) {
  // CHECK_INPUT(points);

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
#endif

const torch::Tensor toSingleFPSPointIdxs(const torch::Tensor &points,
                                   const int &sample_point_num) {
  const torch::TensorOptions opts =
      torch::TensorOptions().dtype(points.dtype()).device(points.device());
  const torch::TensorOptions idx_opts =
      torch::TensorOptions().dtype(torch::kInt64).device(points.device());

  torch::Tensor centroids = torch::zeros({sample_point_num}, idx_opts);
  torch::Tensor distance = torch::ones({points.sizes()[0]}, opts) * 1e10;

  torch::Tensor farthest = torch::zeros({1}, idx_opts);

  for (int i = 0; i < sample_point_num; ++i) {
    centroids.index_put_({i}, farthest);

    const torch::Tensor centroid =
        points.index({farthest, slice_all}).view({1, 3});

    const torch::Tensor point_diffs = points - centroid;

    const torch::Tensor dist = torch::sum(point_diffs * point_diffs, -1);

    const torch::Tensor mask = dist < distance;

    distance.index_put_({mask}, dist.index({mask}));

    farthest = std::get<1>(torch::max(distance, -1));
  }

  return centroids;
}

const torch::Tensor toFPSPointIdxs(const torch::Tensor &points,
                                const torch::Tensor &point_idxs,
                                const float &sample_point_scale,
                                const int &idx_num) {
  const torch::Tensor detach_points = points.detach();

  const torch::Tensor point_counts = toIdxCounts(point_idxs, idx_num);

  const torch::Tensor sample_point_nums =
      torch::ceil(point_counts * sample_point_scale)
          .toType(point_idxs.scalar_type());

#ifdef USE_CUDA
  if (points.is_cuda()) {
    const torch::Tensor fps_point_idxs =
        furthest_point_sampling(detach_points.toType(torch::kFloat32),
                                point_counts.toType(torch::kInt32),
                                sample_point_nums.toType(torch::kInt32));

    return fps_point_idxs;
  }
#endif

  std::vector<torch::Tensor> fps_point_idxs_vec;
  fps_point_idxs_vec.reserve(idx_num);

  std::int64_t point_start_idx = 0;
  for (int i = 0; i < idx_num; ++i) {
    const torch::Tensor current_points = points.index({point_idxs == i});

    const torch::Tensor current_fps_point_idxs = toSingleFPSPointIdxs(
        current_points, sample_point_nums[i].item<std::int64_t>());

    fps_point_idxs_vec.emplace_back(current_fps_point_idxs + point_start_idx);

    point_start_idx += point_counts[i].item<std::int64_t>();
  }

  const torch::Tensor fps_point_idxs = torch::hstack(fps_point_idxs_vec);

  return fps_point_idxs;
}
