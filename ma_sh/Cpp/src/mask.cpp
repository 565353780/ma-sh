#include "mask.h"
#include <torch/script.h>
#include <vector>

using namespace torch::indexing;

const torch::Tensor toMaskBaseValues(const torch::Tensor &phis,
                                     const int &degree_max) {
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

const torch::Tensor toMaskValues(const torch::Tensor &params,
                                 const torch::Tensor &base_values,
                                 const torch::Tensor &phi_idxs) {
  const torch::Tensor repeat_params = params.index({phi_idxs});

  const torch::Tensor values_matrix =
      repeat_params * base_values.transpose(1, 0);

  const torch::Tensor values = torch::sum(values_matrix, 1);

  return values;
}
