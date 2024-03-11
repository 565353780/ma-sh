#include "idx.h"
#include <cstdint>

const torch::Tensor toCounts(const std::vector<torch::Tensor> &data_vec) {
  std::vector<long> counts_vec;
  counts_vec.reserve(data_vec.size());

  for (size_t i = 0; i < data_vec.size(); ++i) {
    counts_vec.emplace_back(data_vec[i].sizes()[0]);
  }

  const torch::TensorOptions opts =
      torch::TensorOptions().dtype(torch::kInt64).device(data_vec[0].device());

  const torch::Tensor counts =
      torch::from_blob(counts_vec.data(), {long(counts_vec.size())}, opts)
          .clone();

  return counts;
}

const torch::Tensor toIdxs(const torch::Tensor &data_counts) {
  std::vector<torch::Tensor> idxs_vec;

  const torch::TensorOptions opts = torch::TensorOptions()
                                        .dtype(data_counts.dtype())
                                        .device(data_counts.device());

  for (int i = 0; i < data_counts.sizes()[0]; ++i) {
    const torch::Tensor current_idxs =
        torch::ones(data_counts[i].item<std::int64_t>(), opts) * i;

    idxs_vec.emplace_back(current_idxs);
  }

  const torch::Tensor idxs = torch::hstack(idxs_vec);

  return idxs;
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
