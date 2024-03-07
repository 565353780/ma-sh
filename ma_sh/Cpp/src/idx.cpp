#include "idx.h"

const torch::Tensor toBoundIdxs(const torch::Tensor &data_counts) {
  torch::Tensor bound_idxs = torch::zeros(data_counts.sizes()[0] + 1)
                                 .toType(data_counts.scalar_type())
                                 .to(data_counts.device());

  for (int i = 1; i < bound_idxs.sizes()[0]; ++i) {
    bound_idxs[i] = data_counts[i - 1] + bound_idxs[i - 1];
  }
  return bound_idxs;
}
