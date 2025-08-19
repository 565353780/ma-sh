#include "filter.h"

using namespace torch::indexing;

const torch::Tensor toMaxValues(const int &unique_idx_num,
                                const torch::Tensor &data,
                                const torch::Tensor &data_idxs) {
  std::vector<torch::Tensor> max_values_vec;
  max_values_vec.reserve(unique_idx_num);

  for (int i = 0; i < unique_idx_num; ++i) {
    const torch::Tensor current_max_value =
        torch::max(data.index({data_idxs == i}));

    max_values_vec.emplace_back(current_max_value);
  }

  const torch::Tensor max_values = torch::hstack(max_values_vec);

  return max_values;
}
