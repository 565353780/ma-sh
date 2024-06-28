#include "simple_mash.h"
#include "constant.h"
#include "direction.h"
#include "fps.h"
#include "mash_unit.h"
#include "mask.h"

const std::vector<torch::Tensor> toSimpleMashSamplePoints(
    const int &anchor_num, const int &mask_degree_max, const int &sh_degree_max,
    const torch::Tensor &mask_params, const torch::Tensor &sh_params,
    const torch::Tensor &rotate_vectors, const torch::Tensor &positions,
    const int &sample_phi_num, const int &sample_theta_num,
    const bool &use_inv) {
  const torch::Tensor mask_boundary_thetas = toMaskBoundaryThetas(
      mask_params, mask_boundary_base_values, mask_boundary_phi_idxs);

  return;
}
