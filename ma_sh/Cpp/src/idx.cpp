#include "idx.h"

using namespace torch::indexing;

const torch::Tensor toBoundIdxs(const torch::Tensor &data_counts) {
  torch::Tensor bound_idxs = torch::zeros(data_counts.sizes()[0] + 1)
                                 .toType(data_counts.scalar_type())
                                 .to(data_counts.device());

  for (int i = 1; i < bound_idxs.sizes()[0]; ++i) {
    bound_idxs[i] = data_counts[i - 1] + bound_idxs[i - 1];
  }

  return bound_idxs;
}

const torch::Tensor toMaskBoundaryPhis(const int &anchor_num,
                                       const int &mask_boundary_sample_num) {
  torch::Tensor mask_boundary_phis =
      torch::zeros({anchor_num, mask_boundary_sample_num});

  const Slice slice_all(None);

  for (int i = 0; i < mask_boundary_sample_num; ++i) {
    const float current_phi = 2.0 * M_PI * i / mask_boundary_sample_num;

    mask_boundary_phis.index_put_({slice_all, i}, current_phi);
  }

  mask_boundary_phis = mask_boundary_phis.reshape({-1});

  return mask_boundary_phis;
}
