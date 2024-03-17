#include "idx.h"
#include "constant.h"
#include <cstdint>

const torch::Tensor toCounts(const std::vector<torch::Tensor> &data_vec) {
  std::vector<torch::Tensor> counts_vec;
  counts_vec.reserve(data_vec.size());

  const torch::TensorOptions opts = torch::TensorOptions()
                                        .dtype(data_vec[0].dtype())
                                        .device(data_vec[0].device());

  for (size_t i = 0; i < data_vec.size(); ++i) {
    counts_vec.emplace_back(torch::tensor(data_vec[i].sizes()[0], opts));
  }

  const torch::Tensor counts = torch::hstack(counts_vec);

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

const torch::Tensor toDataIdxs(const int &repeat_num, const int &idx_num) {
  const torch::TensorOptions opts =
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);

  torch::Tensor data_idxs_matrix = torch::zeros({repeat_num, idx_num}, opts);

  for (int i = 0; i < idx_num; ++i) {
    data_idxs_matrix.index_put_({slice_all, i}, i);
  }

  const torch::Tensor data_idxs = data_idxs_matrix.reshape({-1});

  return data_idxs;
}

const torch::Tensor toIdxCounts(const torch::Tensor &idxs, const int &idx_num) {
  const torch::TensorOptions idx_opts =
      torch::TensorOptions().dtype(idxs.dtype()).device(idxs.device());

  torch::Tensor idx_counts = torch::zeros({idx_num}, idx_opts);

  for (int i = 0; i < idx_num; ++i) {
    const torch::Tensor current_idx_count = torch::sum(idxs == i);

    idx_counts[i] = current_idx_count;
  }

  return idx_counts;
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
