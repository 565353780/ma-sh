#include "sample.h"
#include <c10/core/DeviceType.h>
#include <torch/types.h>

using namespace torch::indexing;

const torch::Tensor toUniformSamplePhis(const int &point_num) {
  std::vector<float> phis_vec;
  phis_vec.reserve(point_num);

  for (int i = 0; i < point_num; ++i) {
    phis_vec[i] = (2.0 * i + 1.0) / point_num - 1.0;
  }

  const torch::TensorOptions opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

  const torch::Tensor phis =
      torch::from_blob(phis_vec.data(), {point_num}, opts).clone();

  return phis;
}

const torch::Tensor toUniformSampleThetas(const torch::Tensor &phis) {
  const float weight = std::sqrt(phis.sizes()[0] * M_PI);

  const torch::Tensor thetas = weight * phis;

  return thetas;
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
