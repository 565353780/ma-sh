#include "mask.h"
#include <torch/script.h>
#include <vector>

using namespace torch::indexing;

const torch::Tensor toMaskBaseValues(const int &degree_max,
                                     const torch::Tensor &phis) {
  std::vector<torch::Tensor> base_values_vec;
  base_values_vec.reserve(2 * degree_max + 1);

  base_values_vec.emplace_back(torch::ones_like(phis));

  for (int i = 1; i < degree_max + 1; ++i) {
    const torch::Tensor current_phis = float(i) * phis;
    base_values_vec.emplace_back(torch::cos(current_phis));
    base_values_vec.emplace_back(torch::sin(current_phis));
  }

  const torch::Tensor base_values = torch::vstack(base_values_vec);

  return base_values;
}

const torch::Tensor toMaskValues(const torch::Tensor &phi_idxs,
                                 const torch::Tensor &params,
                                 const torch::Tensor &base_values) {
  std::vector<torch::Tensor> values_vec;

  const int crop_num = phi_idxs.sizes()[0] - 1;

  values_vec.reserve(crop_num);

  const Slice slice_all(None);

  for (int i = 0; i < crop_num; ++i) {
    const Slice slice_crop =
        Slice(phi_idxs[i].item<int>(), phi_idxs[i + 1].item<int>());

    const torch::Tensor crop_base_values =
        base_values.index({slice_all, slice_crop});

    const torch::Tensor crop_values =
        torch::matmul(params[i], crop_base_values);

    values_vec.emplace_back(crop_values);
  }

  const torch::Tensor values = torch::hstack(values_vec);

  return values;
}
