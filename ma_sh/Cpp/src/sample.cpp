#include "sample.h"
#include "constant.h"
#include <c10/core/DeviceType.h>
#include <cstdint>
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
  torch::Tensor mask_boundary_phi_matrix =
      torch::zeros({anchor_num, mask_boundary_sample_num});

  for (int i = 0; i < mask_boundary_sample_num; ++i) {
    const float current_phi = PI_2 * i / mask_boundary_sample_num;

    mask_boundary_phi_matrix.index_put_({slice_all, i}, current_phi);
  }

  const torch::Tensor mask_boundary_phis =
      mask_boundary_phi_matrix.reshape({-1});

  return mask_boundary_phis;
}

const torch::Tensor toSampleThetaNums(const torch::Tensor &mask_boundary_thetas,
                                      const float &delta_theta_angle) {
  const torch::Tensor detach_mask_boundary_thetas =
      mask_boundary_thetas.detach();

  const float delta_theta = M_PI / 90.0 * delta_theta_angle;

  const torch::Tensor sample_theta_nums =
      torch::ceil(detach_mask_boundary_thetas / delta_theta)
          .toType(torch::kInt64);

  return sample_theta_nums;
}

const torch::Tensor toSampleThetas(const torch::Tensor &mask_boundary_thetas,
                                   const torch::Tensor &sample_theta_nums) {
  std::vector<torch::Tensor> sample_thetas_vec;
  sample_thetas_vec.reserve(sample_theta_nums.size(0));

  const torch::TensorOptions opts = torch::TensorOptions()
                                        .dtype(mask_boundary_thetas.dtype())
                                        .device(mask_boundary_thetas.device());

  for (int i = 0; i < sample_theta_nums.size(0); ++i) {
    const int current_sample_theta_num = sample_theta_nums[i].item<int>();

    const torch::Tensor current_sample_thetas =
        torch::arange(1, current_sample_theta_num + 1, opts) /
        current_sample_theta_num * mask_boundary_thetas[i];

    sample_thetas_vec.emplace_back(current_sample_thetas);
  }

  const torch::Tensor sample_thetas = torch::hstack(sample_thetas_vec);

  return sample_thetas;
}
