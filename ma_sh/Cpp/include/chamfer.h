#pragma once

#include <torch/extension.h>

#ifdef USE_CUDA
int chamfer_cuda_forward(const torch::Tensor &xyz1, const torch::Tensor &xyz2,
                         torch::Tensor &dist1, torch::Tensor &dist2,
                         torch::Tensor &idx1, torch::Tensor &idx2);

int chamfer_cuda_backward(const torch::Tensor &xyz1, const torch::Tensor &xyz2,
                          const torch::Tensor &graddist1,
                          const torch::Tensor &graddist2,
                          const torch::Tensor &idx1, const torch::Tensor &idx2,
                          torch::Tensor &gradxyz1, torch::Tensor &gradxyz2);

class chamfer_3DFunction
    : public torch::autograd::Function<chamfer_3DFunction> {
public:
  static std::vector<torch::Tensor>
  forward(torch::autograd::AutogradContext *ctx, const torch::Tensor &xyz1,
          const torch::Tensor &xyz2);

  static std::vector<torch::Tensor>
  backward(torch::autograd::AutogradContext *ctx,
           std::vector<torch::Tensor> &grad_outputs);
};

const std::vector<torch::Tensor> chamfer_3DDist(const torch::Tensor &input1,
                                                const torch::Tensor &input2);
#endif

const torch::Tensor batched_pairwise_dist(const torch::Tensor &x,
                                          const torch::Tensor &y);

const std::vector<torch::Tensor> distChamfer(const torch::Tensor &a,
                                             const torch::Tensor &b);

const std::vector<torch::Tensor> toChamferDistance(const torch::Tensor &a,
                                                   const torch::Tensor &b);
