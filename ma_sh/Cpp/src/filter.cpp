#include "filter.h"

using namespace torch::indexing;

const torch::Tensor
toMaskBoundaryMaxThetas(const torch::Tensor &mask_boundary_thetas,
                        const torch::Tensor &mask_boundary_phi_idxs) {
  const int anchor_num = mask_boundary_phi_idxs.sizes()[0] - 1;

  std::vector<torch::Tensor> max_thetas_vec;
  max_thetas_vec.reserve(anchor_num);

  for (int i = 0; i < anchor_num; ++i) {
    const Slice slice_crop = Slice(mask_boundary_phi_idxs[i].item<int>(),
                                   mask_boundary_phi_idxs[i + 1].item<int>());

    const torch::Tensor current_max_phi =
        torch::max(mask_boundary_thetas.index({slice_crop}));

    max_thetas_vec.emplace_back(current_max_phi);
  }

  const torch::Tensor max_thetas = torch::hstack(max_thetas_vec);

  return max_thetas;
}
