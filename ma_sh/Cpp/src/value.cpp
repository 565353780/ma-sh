#include "value.h"
#include <torch/script.h>

const torch::Tensor toValues(const torch::Tensor &params,
                             const torch::Tensor &base_values,
                             const torch::Tensor &phi_idxs) {
  const torch::Tensor repeat_params = params.index({phi_idxs});

  const torch::Tensor values_matrix =
      repeat_params * base_values.transpose(1, 0);

  const torch::Tensor values = torch::sum(values_matrix, 1);

  return values;
}
