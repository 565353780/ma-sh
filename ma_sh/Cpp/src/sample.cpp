#include "sample.h"
#include <cmath>

const torch::Tensor getUniformSamplePhis(const int &point_num,
                                         const torch::Dtype &dtype,
                                         const torch::Device &device) {
  std::vector<float> phis_vec;
  phis_vec.reserve(point_num);

  for (int i = 0; i < point_num; ++i) {
    phis_vec[i] = (2.0 * i + 1.0) / point_num - 1.0;
  }

  const torch::Tensor phis =
      torch::from_blob(phis_vec.data(), {point_num}, dtype).clone().to(device);

  return phis;
}

const torch::Tensor getUniformSampleThetas(const torch::Tensor &phis) {
  const torch::Tensor thetas = std::sqrt(phis.sizes()[0] * M_PI) * phis;

  return thetas;
}
