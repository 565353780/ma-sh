#include "sampling.h"
#include "utils.h"

void furthest_point_sampling_kernel_wrapper(int b, int n, int m,
                                            const float *dataset, float *temp,
                                            int *idxs);

const torch::Tensor furthest_point_sampling(const torch::Tensor &points,
                                            const int &nsamples) {
  CHECK_CONTIGUOUS(points);
  CHECK_IS_FLOAT(points);

  const torch::TensorOptions float_opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(points.device());

  const torch::TensorOptions idx_opts =
      torch::TensorOptions().dtype(torch::kInt32).device(points.device());

  torch::Tensor output = torch::zeros({points.size(0), nsamples}, idx_opts);

  torch::Tensor tmp =
      torch::full({points.size(0), points.size(1)}, 1e10, float_opts);

  if (points.is_cuda()) {
    furthest_point_sampling_kernel_wrapper(
        points.size(0), points.size(1), nsamples, points.data_ptr<float>(),
        tmp.data_ptr<float>(), output.data_ptr<int>());
  } else {
    AT_ASSERT(false, "CPU not supported");
  }

  return output;
}
