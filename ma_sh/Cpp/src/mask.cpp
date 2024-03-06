#include "mask.h"
#include <vector>

const torch::Tensor getMaskBaseValues(const int &degree_max,
                                      const torch::Tensor &phis) {
  std::vector<torch::Tensor> base_values;
  base_values.reserve(2 * degree_max + 1);

  base_values.emplace_back(torch::ones_like(phis));

  for (int i = 1; i < degree_max + 1; ++i) {
    const torch::Tensor current_phis = float(i) * phis;
    base_values.emplace_back(torch::cos(current_phis));
    base_values.emplace_back(torch::sin(current_phis));
  }

  return torch::vstack(base_values);
}
