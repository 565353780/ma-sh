#include "sample.h"
#include "constant.h"
#include <c10/core/DeviceType.h>
#include <torch/types.h>

using namespace torch::indexing;

const torch::Tensor toUniformSamplePhis(const int &sample_num) {
  std::vector<float> phis_vec;
  phis_vec.reserve(sample_num);

  for (int i = 0; i < sample_num; ++i) {
    phis_vec.emplace_back(PHI_WEIGHT * (i + 0.5));
  }

  const torch::TensorOptions opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

  const torch::Tensor phis =
      torch::from_blob(phis_vec.data(), {sample_num}, opts).clone();

  return phis;
}

const torch::Tensor toUniformSampleThetas(const int &sample_num) {
  std::vector<float> cos_thetas_vec;
  cos_thetas_vec.reserve(sample_num);

  for (int i = 0; i < sample_num; ++i) {
    cos_thetas_vec[i] = 1.0 - (2.0 * i + 1.0) / sample_num;
  }

  const torch::TensorOptions opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

  const torch::Tensor cos_thetas =
      torch::from_blob(cos_thetas_vec.data(), {sample_num}, opts).clone();

  const torch::Tensor thetas = torch::acos(cos_thetas);

  return thetas;
}

const torch::Tensor toMaskBoundaryPhis(const int &anchor_num,
                                       const int &mask_boundary_sample_num) {
  torch::Tensor mask_boundary_phis =
      torch::zeros({anchor_num, mask_boundary_sample_num});

  const Slice slice_all(None);

  for (int i = 0; i < mask_boundary_sample_num; ++i) {
    const float current_phi = PI_2 * i / mask_boundary_sample_num;

    mask_boundary_phis.index_put_({slice_all, i}, current_phi);
  }

  mask_boundary_phis = mask_boundary_phis.reshape({-1});

  return mask_boundary_phis;
}
