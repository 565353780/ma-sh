#include "single_ops.h"
#include "mash_unit.h"
#include "mask.h"
#include "rotate.h"

const torch::Tensor toSingleRotateMatrix(const torch::Tensor &rotate_vector) {
  const torch::Tensor v_detach_rotate_vector =
      rotate_vector.detach().reshape({1, -1});

  const torch::Tensor v_single_rotate_matrix =
      toRotateMatrixs(v_detach_rotate_vector);

  const torch::Tensor single_rotate_matrix = v_single_rotate_matrix[0];

  return single_rotate_matrix;
}

const torch::Tensor toSingleMaskBoundaryThetas(const int &mask_degree_max,
                                               const torch::Tensor &mask_param,
                                               const torch::Tensor &mask_phis) {
  const torch::Tensor h_detach_mask_param =
      mask_param.detach().reshape({1, -1});

  const torch::Tensor mask_base_values =
      toMaskBaseValues(mask_phis, mask_degree_max);

  const torch::Tensor single_mask_boundary_thetas =
      toMaskBoundaryThetas(h_detach_mask_param, mask_base_values,
                           torch::zeros_like(mask_phis).toType(torch::kInt64));

  return single_mask_boundary_thetas;
}

const torch::Tensor toSingleSHDists(const int &sh_degree_max,
                                    const torch::Tensor &sh_param,
                                    const torch::Tensor &sh_phis,
                                    const torch::Tensor &sh_thetas) {
  const torch::Tensor v_detach_sh_param = sh_param.detach().reshape({1, -1});

  const torch::Tensor sh_values =
      toSHValues(sh_degree_max, v_detach_sh_param, sh_phis, sh_thetas,
                 torch::zeros_like(sh_phis).toType(torch::kInt64));

  return sh_values;
}
