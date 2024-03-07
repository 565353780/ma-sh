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

const std::vector<torch::Tensor>
toInMaskSamplePolarIdxs(const torch::Tensor &sample_thetas,
                        const torch::Tensor &mask_boundary_max_thetas) {
  std::vector<torch::Tensor> in_mask_idxs_vec;
  in_mask_idxs_vec.reserve(mask_boundary_max_thetas.sizes()[0]);

  for (int i = 0; i < mask_boundary_max_thetas.sizes()[0]; ++i) {
    const torch::Tensor current_in_mask_idxs =
        torch::where(sample_thetas <= mask_boundary_max_thetas[i])[0];

    in_mask_idxs_vec.emplace_back(current_in_mask_idxs);
  }

  return in_mask_idxs_vec;
}
