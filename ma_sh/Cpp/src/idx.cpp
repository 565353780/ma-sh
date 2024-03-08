#include "idx.h"

const torch::Tensor toCounts(const std::vector<torch::Tensor> &data_vec) {
  std::vector<long> counts_vec;
  counts_vec.reserve(data_vec.size());

  for (size_t i = 0; i < data_vec.size(); ++i) {
    counts_vec.emplace_back(data_vec[i].sizes()[0]);
  }

  const torch::TensorOptions opts = torch::TensorOptions()
                                        .dtype(data_vec[0].dtype())
                                        .device(data_vec[0].device());

  const torch::Tensor counts =
      torch::from_blob(counts_vec.data(), {long(counts_vec.size())}, opts)
          .clone();

  return counts;
}

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
toLowerIdxsVec(const torch::Tensor &values, const torch::Tensor &max_bounds) {
  std::vector<torch::Tensor> lower_idxs_vec;
  lower_idxs_vec.reserve(max_bounds.sizes()[0]);

  for (int i = 0; i < max_bounds.sizes()[0]; ++i) {
    const torch::Tensor current_lower_idxs =
        torch::where(values <= max_bounds[i])[0];

    lower_idxs_vec.emplace_back(current_lower_idxs);
  }

  return lower_idxs_vec;
}
