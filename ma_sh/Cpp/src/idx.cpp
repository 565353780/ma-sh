#include "idx.h"
#include "constant.h"

using namespace torch::indexing;

const torch::Tensor toCounts(const std::vector<torch::Tensor> &data_vec) {
  const torch::TensorOptions opts = torch::TensorOptions()
                                        .dtype(data_vec[0].dtype())
                                        .device(data_vec[0].device());

  torch::Tensor counts = torch::zeros({long(data_vec.size())}, opts);

  for (size_t i = 0; i < data_vec.size(); ++i) {
    counts[i] = data_vec[i].size(0);
  }

  return counts;
}

const torch::Tensor toIdxs(const torch::Tensor &data_counts) {
  const int idxs_num = torch::sum(data_counts).item<int>();

  const torch::TensorOptions opts = torch::TensorOptions()
                                        .dtype(data_counts.dtype())
                                        .device(data_counts.device());

  torch::Tensor idxs = torch::zeros({idxs_num}, opts);

  int idx_start_idx = 0;
  for (int i = 0; i < data_counts.size(0); ++i) {
    const int current_data_count = data_counts[i].item<int>();

    const Slice current_slice =
        Slice(idx_start_idx, idx_start_idx + current_data_count);

    idxs.index_put_({current_slice}, i);

    idx_start_idx += current_data_count;
  }

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
  lower_idxs_vec.reserve(max_bounds.size(0));

  for (int i = 0; i < max_bounds.size(0); ++i) {
    const torch::Tensor current_lower_idxs =
        torch::where(values <= max_bounds[i])[0];

    lower_idxs_vec.emplace_back(current_lower_idxs);
  }

  return lower_idxs_vec;
}
