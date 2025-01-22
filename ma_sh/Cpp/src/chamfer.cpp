#include "chamfer.h"

const torch::Tensor toValidTensor(const torch::Tensor &source_tensor) {
  torch::Tensor valid_tensor = source_tensor;

  const torch::Tensor nan_mask = torch::isnan(source_tensor);

  valid_tensor.masked_fill_(nan_mask, 0.0);

  return valid_tensor;
}

#ifdef USE_CUDA
std::vector<torch::Tensor>
chamfer_3DFunction::forward(torch::autograd::AutogradContext *ctx,
                            const torch::Tensor &xyz1,
                            const torch::Tensor &xyz2) {
  int batchsize = xyz1.size(0);
  int n = xyz1.size(1);
  int m = xyz2.size(1);

  torch::TensorOptions opts =
      torch::TensorOptions().dtype(xyz1.dtype()).device(xyz1.device());

  torch::TensorOptions idx_opts =
      torch::TensorOptions().dtype(torch::kInt).device(xyz1.device());

  torch::Tensor dist1 = torch::zeros({batchsize, n}, opts);
  torch::Tensor dist2 = torch::zeros({batchsize, m}, opts);

  torch::Tensor idx1 = torch::zeros({batchsize, n}, idx_opts);
  torch::Tensor idx2 = torch::zeros({batchsize, m}, idx_opts);

  chamfer_cuda_forward(xyz1, xyz2, dist1, dist2, idx1, idx2);

  ctx->save_for_backward({xyz1, xyz2, idx1, idx2});

  std::vector<torch::Tensor> dists_with_idxs({dist1, dist2, idx1, idx2});

  return dists_with_idxs;
}

std::vector<torch::Tensor>
chamfer_3DFunction::backward(torch::autograd::AutogradContext *ctx,
                             std::vector<torch::Tensor> &grad_outputs) {
  std::vector<torch::Tensor> dists_with_idxs = ctx->get_saved_variables();

  torch::Tensor xyz1 = dists_with_idxs[0];
  torch::Tensor xyz2 = dists_with_idxs[1];
  torch::Tensor idx1 = dists_with_idxs[2];
  torch::Tensor idx2 = dists_with_idxs[3];

  torch::Tensor &graddist1 = grad_outputs[0];
  torch::Tensor &graddist2 = grad_outputs[1];

  torch::Tensor contiguous_graddist1 = graddist1.contiguous();
  torch::Tensor contiguous_graddist2 = graddist2.contiguous();

  torch::Tensor gradxyz1 = torch::zeros_like(xyz1);
  torch::Tensor gradxyz2 = torch::zeros_like(xyz2);

  chamfer_cuda_backward(xyz1, xyz2, graddist1, graddist2, idx1, idx2, gradxyz1,
                        gradxyz2);

  std::vector<torch::Tensor> grads({gradxyz1, gradxyz2});

  return grads;
}

const std::vector<torch::Tensor> chamfer_3DDist(const torch::Tensor &input1,
                                                const torch::Tensor &input2) {
  const torch::Tensor contiguous_input1 = input1.contiguous();
  const torch::Tensor contiguous_input2 = input2.contiguous();

  const std::vector<torch::Tensor> dists_with_idxs =
      chamfer_3DFunction::apply(contiguous_input1, contiguous_input2);

  const torch::Tensor valid_dists1 = toValidTensor(dists_with_idxs[0]);
  const torch::Tensor valid_dists2 = toValidTensor(dists_with_idxs[1]);

  const std::vector<torch::Tensor> valid_dists_with_idxs(
      {valid_dists1, valid_dists2, dists_with_idxs[2], dists_with_idxs[3]});

  return valid_dists_with_idxs;
}
#endif

const torch::Tensor batched_pairwise_dist(const torch::Tensor &x,
                                          const torch::Tensor &y) {
  int bs = x.size(0);
  int num_points_x = x.size(1);
  int num_points_y = y.size(1);

  const torch::Tensor xx = torch::pow(x, 2).sum(2);
  const torch::Tensor yy = torch::pow(y, 2).sum(2);
  torch::Tensor zz;
  if (num_points_x < num_points_y) {
    zz = torch::bmm(2.0f * x, y.transpose(2, 1));
  } else {
    zz = torch::bmm(x, (2.0f * y).transpose(2, 1));
  }

  const torch::Tensor rx =
      xx.unsqueeze(2).expand({bs, num_points_x, num_points_y});
  const torch::Tensor ry =
      yy.unsqueeze(1).expand({bs, num_points_x, num_points_y});

  const torch::Tensor P = rx + ry - zz;

  return P;
}

const std::vector<torch::Tensor> distChamfer(const torch::Tensor &a,
                                             const torch::Tensor &b) {
  const torch::Tensor P = batched_pairwise_dist(a, b);

  const std::tuple<torch::Tensor, torch::Tensor> P1 = torch::min(P, 2);
  const std::tuple<torch::Tensor, torch::Tensor> P2 = torch::min(P, 1);

  const torch::Tensor &dists1 = std::get<0>(P1);
  const torch::Tensor &dists2 = std::get<0>(P2);
  const torch::Tensor idxs1 = std::get<1>(P1).toType(torch::kInt);
  const torch::Tensor idxs2 = std::get<1>(P2).toType(torch::kInt);

  const torch::Tensor valid_dists1 = toValidTensor(dists1);
  const torch::Tensor valid_dists2 = toValidTensor(dists2);

  const std::vector<torch::Tensor> dists_with_idxs(
      {valid_dists1, valid_dists2, idxs1, idxs2});

  return dists_with_idxs;
}

const std::vector<torch::Tensor> toChamferDistance(const torch::Tensor &a,
                                                   const torch::Tensor &b) {
#ifdef USE_CUDA
  if (a.is_cuda()) {
    return chamfer_3DDist(a, b);
  }
#endif

  return distChamfer(a, b);
}
