#include "idx.h"
#include <c10/core/DeviceType.h>
#include <torch/types.h>

const torch::Tensor toBoundIdxs(const torch::Tensor &data_counts) {
  torch::Tensor bound_idxs = torch::zeros(data_counts.sizes()[0] + 1)
                                 .toType(data_counts.scalar_type())
                                 .to(data_counts.device());

  for (int i = 1; i < bound_idxs.sizes()[0]; ++i) {
    bound_idxs[i] = data_counts[i - 1] + bound_idxs[i - 1];
  }

  return bound_idxs;
}

const std::vector<torch::Tensor>
toLowerValueIdxsVec(const torch::Tensor &values,
                    const torch::Tensor &max_bounds) {
  std::vector<torch::Tensor> lower_value_idxs_vec;
  lower_value_idxs_vec.reserve(max_bounds.sizes()[0]);

  for (int i = 0; i < max_bounds.sizes()[0]; ++i) {
    const torch::Tensor current_lower_value_idxs =
        torch::where(values <= max_bounds[i])[0];

    lower_value_idxs_vec.emplace_back(current_lower_value_idxs);
  }

  return lower_value_idxs_vec;
}

const torch::Tensor toInMaskSamplePolarCounts(
    const std::vector<torch::Tensor> &in_mask_sample_polar_idxs_vec) {
  std::vector<int> in_mask_sample_polar_counts_vec;
  in_mask_sample_polar_counts_vec.reserve(in_mask_sample_polar_idxs_vec.size());

  for (size_t i = 0; i < in_mask_sample_polar_idxs_vec.size(); ++i) {
    in_mask_sample_polar_counts_vec[i] =
        in_mask_sample_polar_idxs_vec[i].sizes()[0];
  }

  const torch::TensorOptions opts =
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);

  const torch::Tensor in_mask_sample_polar_counts =
      torch::from_blob(in_mask_sample_polar_counts_vec.data(),
                       {long(in_mask_sample_polar_counts_vec.size())}, opts)
          .clone();

  return in_mask_sample_polar_counts;
}
