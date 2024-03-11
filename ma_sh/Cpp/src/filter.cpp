#include "filter.h"
#include <ATen/ops/_unique.h>

using namespace torch::indexing;

const torch::Tensor toMaxValues(const torch::Tensor &data,
                                const torch::Tensor &data_idxs) {
  const torch::Tensor unique_idxs = std::get<0>(at::_unique(data_idxs));
  const int idx_num = unique_idxs.sizes()[0];

  std::vector<torch::Tensor> max_values_vec;
  max_values_vec.reserve(idx_num);

  for (int i = 0; i < idx_num; ++i) {
    const torch::Tensor current_max_value =
        torch::max(data.index({data_idxs == unique_idxs[i]}));

    max_values_vec.emplace_back(current_max_value);
  }

  const torch::Tensor max_values = torch::hstack(max_values_vec);

  return max_values;
}
