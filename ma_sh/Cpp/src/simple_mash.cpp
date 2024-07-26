#include "simple_mash.h"
#include "idx.h"
#include "mash.h"
#include "mash_unit.h"
#include "timer.h"

using namespace torch::indexing;

const torch::Tensor toSimpleMashSamplePoints(
    const int &anchor_num, const int &mask_degree_max, const int &sh_degree_max,
    const torch::Tensor &mask_params, const torch::Tensor &sh_params,
    const torch::Tensor &rotate_vectors, const torch::Tensor &positions,
    const torch::Tensor &sample_phis, const torch::Tensor &sample_base_values,
    const int &sample_theta_num, const bool &use_inv) {
  Timer timer;
  const torch::TensorOptions idx_opts =
      torch::TensorOptions().dtype(torch::kInt64).device(sample_phis.device());

  const torch::TensorOptions opts = torch::TensorOptions()
                                        .dtype(sample_phis.dtype())
                                        .device(sample_phis.device());

  const torch::Tensor single_anchor_sample_phis = torch::hstack(
      {sample_phis[0], sample_phis.repeat({1, sample_theta_num}).reshape(-1)});

  const torch::Tensor repeat_sample_base_values =
      sample_base_values.repeat({1, anchor_num});

  const int64_t sample_phi_num = sample_phis.size(0);

  timer.reset();
  const torch::Tensor sample_phi_counts =
      torch::ones({anchor_num}, idx_opts) * sample_phi_num;

  const torch::Tensor sample_phi_idxs = toIdxs(sample_phi_counts);

  const torch::Tensor mask_boundary_thetas = toMaskBoundaryThetas(
      mask_params, repeat_sample_base_values, sample_phi_idxs);
  std::cout << "toMaskBoundaryThetas : " << timer.now() << std::endl;

  timer.reset();
  const torch::Tensor single_anchor_sample_theta_idxs = torch::hstack(
      {torch::zeros({1}, idx_opts), torch::arange(sample_phi_num, idx_opts)
                                        .repeat({sample_theta_num, 1})
                                        .reshape(-1)});

  const int64_t single_anchor_sample_theta_num =
      single_anchor_sample_theta_idxs.size(0);

  torch::Tensor full_sample_theta_idxs =
      torch::zeros({single_anchor_sample_theta_num * anchor_num}, idx_opts);

  for (int i = 0; i < anchor_num; ++i) {
    const Slice current_anchor_sample_theta_slice =
        Slice(i * single_anchor_sample_theta_num,
              (i + 1) * single_anchor_sample_theta_num);

    full_sample_theta_idxs.index_put_({current_anchor_sample_theta_slice},
                                      single_anchor_sample_theta_idxs +
                                          sample_phi_num);
  }
  std::cout << "create full_sample_theta_idxs : " << timer.now() << std::endl;

  const torch::Tensor single_cycle_sample_theta_weights =
      torch::arange(1, sample_theta_num + 1, opts) / sample_theta_num;

  const torch::Tensor single_anchor_sample_theta_weights =
      torch::hstack({torch::zeros({1}, opts), single_cycle_sample_theta_weights
                                                  .repeat({sample_phi_num, 1})
                                                  .permute({1, 0})
                                                  .reshape(-1)});

  const torch::Tensor full_sample_phis =
      single_anchor_sample_phis.repeat({anchor_num});

  const torch::Tensor full_sample_theta_weights =
      single_anchor_sample_theta_weights.repeat({anchor_num});

  const torch::Tensor full_sample_thetas =
      mask_boundary_thetas.index({full_sample_theta_idxs}) *
      full_sample_theta_weights;

  const torch::Tensor full_sample_polar_idxs =
      torch::arange(anchor_num, idx_opts)
          .repeat({single_anchor_sample_phis.size(0), 1})
          .permute({1, 0})
          .reshape(-1);

  timer.reset();
  const torch::Tensor sample_points = toSamplePoints(
      mask_degree_max, sh_degree_max, sh_params, rotate_vectors, positions,
      full_sample_phis, full_sample_thetas, full_sample_polar_idxs, use_inv);
  std::cout << "toSamplePoints : " << timer.now() << std::endl;

  return sample_points;
}
