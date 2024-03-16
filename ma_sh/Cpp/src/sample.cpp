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

const torch::Tensor toFPSPointIdxs(const torch::Tensor &points,
                                   const int &sample_point_num) {
  const torch::TensorOptions opts =
      torch::TensorOptions().dtype(points.dtype()).device(points.device());
  const torch::TensorOptions idx_opts =
      torch::TensorOptions().dtype(torch::kInt64).device(points.device());

  torch::Tensor centroids = torch::zeros({sample_point_num}, idx_opts);
  torch::Tensor distance = torch::ones({points.sizes()[0]}, opts) * 1e10;

  torch::Tensor farthest = torch::randint(0, points.sizes()[0], {1}, idx_opts);

  for (int i = 0; i < sample_point_num; ++i) {
    centroids.index_put_({i}, farthest);

    const torch::Tensor centroid =
        points.index({farthest, slice_all}).view({1, 3});

    const torch::Tensor point_diffs = points - centroid;

    const torch::Tensor dist = torch::sum(point_diffs * point_diffs, -1);

    const torch::Tensor mask = dist < distance;

    distance.index_put_({mask}, dist.index({mask}));

    farthest = std::get<1>(torch::max(distance, -1));
  }

  return centroids;
}
