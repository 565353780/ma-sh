#include "rotate.h"
#include "constant.h"
#include <iostream>

using namespace torch::indexing;

const torch::Tensor toRotateMatrixs(const torch::Tensor &rotate_vectors) {
  const torch::Tensor thetas = torch::norm(rotate_vectors, 2, 1);

  const torch::Tensor valid_theta_mask = thetas > 0.0;

  const torch::TensorOptions opts = torch::TensorOptions()
                                        .dtype(rotate_vectors.dtype())
                                        .device(rotate_vectors.device());

  torch::Tensor divide_thetas = torch::ones({rotate_vectors.size(0)}, opts);

  divide_thetas.index_put_({valid_theta_mask},
                           thetas.index({valid_theta_mask}));

  const torch::Tensor v_divide_thetas = divide_thetas.reshape({-1, 1});

  const torch::Tensor normed_rotate_vectors = rotate_vectors / v_divide_thetas;

  torch::Tensor theta_hats = torch::zeros({rotate_vectors.size(0), 3, 3}, opts);

  theta_hats.index_put_({slice_all, 0, 1},
                        -1.0 * normed_rotate_vectors.index({slice_all, 2}));
  theta_hats.index_put_({slice_all, 0, 2},
                        normed_rotate_vectors.index({slice_all, 1}));
  theta_hats.index_put_({slice_all, 1, 0},
                        normed_rotate_vectors.index({slice_all, 2}));
  theta_hats.index_put_({slice_all, 1, 2},
                        -1.0 * normed_rotate_vectors.index({slice_all, 0}));
  theta_hats.index_put_({slice_all, 2, 0},
                        -1.0 * normed_rotate_vectors.index({slice_all, 1}));
  theta_hats.index_put_({slice_all, 2, 1},
                        normed_rotate_vectors.index({slice_all, 0}));

  const torch::Tensor identity_matrix = torch::eye(3, opts);

  const torch::Tensor identity_matrixs =
      identity_matrix.repeat({rotate_vectors.size(0), 1, 1});

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

const torch::Tensor
toRotateVectorsByFaceForwardVectors(const torch::Tensor face_forward_vectors) {
  torch::Tensor rotate_vectors = torch::zeros_like(face_forward_vectors);

  const torch::Tensor face_forward_vector_norms =
      torch::norm(face_forward_vectors, 2, 1);

  const torch::Tensor valid_face_forward_vector_mask =
      face_forward_vector_norms > 0;

  const torch::Tensor valid_face_forward_vectors =
      face_forward_vectors.index({valid_face_forward_vector_mask});

  const torch::Tensor valid_face_forward_vector_norms =
      face_forward_vector_norms.index({valid_face_forward_vector_mask});

  const torch::Tensor v_valid_face_forward_vector_norms =
      valid_face_forward_vector_norms.reshape({-1, 1});

  const torch::Tensor valid_normed_face_forward_vectors =
      valid_face_forward_vectors / v_valid_face_forward_vector_norms;

  torch::Tensor valid_z_axis =
      torch::zeros_like(valid_normed_face_forward_vectors);

  valid_z_axis.index_put_({slice_all, 2}, 1.0);

  torch::Tensor valid_rotate_vectors =
      torch::cross(valid_z_axis, valid_normed_face_forward_vectors, 1);

  torch::Tensor valid_rotate_vector_norms =
      torch::norm(valid_rotate_vectors, 2, 1);

  const torch::Tensor zero_valid_rotate_vector_mask =
      valid_rotate_vector_norms == 0;

  valid_rotate_vector_norms.index_put_({zero_valid_rotate_vector_mask}, 1.0);

  const torch::Tensor v_valid_rotate_vector_norms =
      valid_rotate_vector_norms.reshape({-1, 1});

  torch::Tensor normed_valid_rotate_vectors =
      valid_rotate_vectors / v_valid_rotate_vector_norms;

  normed_valid_rotate_vectors.index_put_({zero_valid_rotate_vector_mask, 0},
                                         1.0);

  const torch::Tensor valid_cos_thetas =
      valid_normed_face_forward_vectors.index({slice_all, 2});

  const torch::Tensor valid_thetas = torch::acos(valid_cos_thetas);

  const torch::Tensor v_valid_thetas = valid_thetas.reshape({-1, 1});

  rotate_vectors.index_put_({valid_face_forward_vector_mask},
                            normed_valid_rotate_vectors * v_valid_thetas);

  return rotate_vectors;
}

const torch::Tensor toRotateVectors(const torch::Tensor &rotate_matrixs) {
  std::vector<torch::Tensor> traces_vec;
  traces_vec.reserve(rotate_matrixs.size(0));

  for (int i = 0; i < rotate_matrixs.size(0); ++i) {
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

  torch::Tensor divide_sin_thetas = torch::ones({rotate_matrixs.size(0)}, opts);

  divide_sin_thetas.index_put_({valid_sin_thetas_mask},
                               sin_thetas.index({valid_sin_thetas_mask}));

  const torch::Tensor vv_divide_sin_thetas =
      divide_sin_thetas.reshape({-1, 1, 1});

  const torch::Tensor rights =
      0.25 * (rotate_matrixs - rotate_matrixs.permute({0, 2, 1})) /
      vv_divide_sin_thetas;

  torch::Tensor normed_rotate_vectors =
      torch::zeros({rotate_matrixs.size(0), 3}, opts);

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
