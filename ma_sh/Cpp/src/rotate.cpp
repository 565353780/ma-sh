#include "rotate.h"
#include "constant.h"

using namespace torch::indexing;

const torch::Tensor toRotateMatrixs(const torch::Tensor &rotate_vectors) {
  const torch::Tensor thetas = torch::norm(rotate_vectors, 2, 1);

  const torch::Tensor valid_theta_mask = thetas > 0.0;

  const torch::TensorOptions opts = torch::TensorOptions()
                                        .dtype(rotate_vectors.dtype())
                                        .device(rotate_vectors.device());

  torch::Tensor divide_thetas = torch::ones({rotate_vectors.sizes()[0]}, opts);

  divide_thetas.index_put_({valid_theta_mask},
                           thetas.index({valid_theta_mask}));

  const torch::Tensor v_divide_thetas = divide_thetas.reshape({-1, 1});

  const torch::Tensor normed_rotate_vectors = rotate_vectors / v_divide_thetas;

  torch::Tensor theta_hats =
      torch::zeros({rotate_vectors.sizes()[0], 3, 3}, opts);

  theta_hats.index_put_({slice_all, 0, 1},
                        -1.0 * normed_rotate_vectors.index({slice_all, 2}));
  theta_hats.index_put_({slice_all, 0, 2},
                        normed_rotate_vectors.index({slice_all, 1}));
  theta_hats.index_put_({slice_all, 1, 0},
                        normed_rotate_vectors.index({slice_all, 2}));
  theta_hats.index_put_({slice_all, 1, 2},
                        -1.0 * normed_rotate_vectors.index({slice_all, 0}));
  theta_hats.index_put_({slice_all, 2, 1},
                        -1.0 * normed_rotate_vectors.index({slice_all, 0}));
  theta_hats.index_put_({slice_all, 2, 1},
                        normed_rotate_vectors.index({slice_all, 0}));

  const torch::Tensor identity_matrix = torch::eye(3, opts);

  const torch::Tensor identity_matrixs =
      identity_matrix.repeat({rotate_vectors.sizes()[0], 1, 1});

  const torch::Tensor vv_thetas = thetas.reshape({-1, 1, 1});

  const torch::Tensor cos_vv_thetas = torch::cos(vv_thetas);
  const torch::Tensor sin_vv_thetas = torch::sin(vv_thetas);

  const torch::Tensor v_normed_rotate_vectors =
      normed_rotate_vectors.reshape({-1, 3, 1});
  const torch::Tensor h_normed_rotate_vectors =
      normed_rotate_vectors.reshape({-1, 1, 3});

  const torch::Tensor n_nts =
      torch::matmul(v_normed_rotate_vectors, h_normed_rotate_vectors);

  const torch::Tensor rotate_matrixs = cos_vv_thetas * identity_matrixs +
                                       (1.0 - cos_vv_thetas) * n_nts +
                                       sin_vv_thetas * theta_hats;

  return rotate_matrixs;
}

const torch::Tensor toRotateVectors(const torch::Tensor &rotate_matrixs) {
  std::vector<torch::Tensor> traces_vec;
  traces_vec.reserve(rotate_matrixs.sizes()[0]);

  for (int i = 0; i < rotate_matrixs.sizes()[0]; ++i) {
    const torch::Tensor current_traces = torch::trace(rotate_matrixs[i]);

    traces_vec.emplace_back(current_traces);
  }

  const torch::Tensor traces = torch::hstack(traces_vec);

  const torch::Tensor thetas = torch::acos((traces - 1.0) * 0.5);

  const torch::Tensor sin_thetas = torch::sin(thetas);

  const torch::Tensor valid_sin_thetas_mask = sin_thetas != 0.0;

  const torch::TensorOptions opts = torch::TensorOptions()
                                        .dtype(rotate_matrixs.dtype())
                                        .device(rotate_matrixs.device());

  torch::Tensor divide_sin_thetas =
      torch::ones({rotate_matrixs.sizes()[0]}, opts);

  divide_sin_thetas.index_put_({valid_sin_thetas_mask},
                               sin_thetas.index({valid_sin_thetas_mask}));

  const torch::Tensor vv_divide_sin_thetas =
      divide_sin_thetas.reshape({-1, 1, 1});

  const torch::Tensor rights =
      0.25 * (rotate_matrixs - rotate_matrixs.permute({0, 2, 1})) /
      vv_divide_sin_thetas;

  torch::Tensor normed_rotate_vectors =
      torch::zeros({rotate_matrixs.sizes()[0], 3}, opts);

  normed_rotate_vectors.index_put_({slice_all, 0},
                                   rights.index({slice_all, 2, 1}) -
                                       rights.index({slice_all, 1, 2}));
  normed_rotate_vectors.index_put_({slice_all, 1},
                                   rights.index({slice_all, 0, 2}) -
                                       rights.index({slice_all, 2, 0}));
  normed_rotate_vectors.index_put_({slice_all, 2},
                                   rights.index({slice_all, 1, 0}) -
                                       rights.index({slice_all, 0, 1}));

  const torch::Tensor v_thetas = thetas.reshape({-1, 1});

  const torch::Tensor rotate_vectors = normed_rotate_vectors * v_thetas;

  return rotate_vectors;
}
