#include "sample.h"
#include <c10/core/DeviceType.h>
#include <torch/types.h>

const torch::Tensor getUniformSamplePhis(const int &point_num) {
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

const torch::Tensor getUniformSampleThetas(const torch::Tensor &phis) {
  const float weight = std::sqrt(phis.sizes()[0] * M_PI);

  const torch::Tensor thetas = weight * phis;

  return thetas;
}
