#include "direction.h"
#include "constant.h"

const torch::Tensor toDirections(const torch::Tensor &phis,
                                 const torch::Tensor &thetas) {
  const torch::Tensor cos_phis = torch::cos(phis);
  const torch::Tensor sin_phis = torch::sin(phis);
  const torch::Tensor cos_thetas = torch::cos(thetas);
  const torch::Tensor sin_thetas = torch::sin(thetas);

  const torch::Tensor sh_directions_x = cos_phis * sin_thetas;
  const torch::Tensor sh_directions_y = sin_phis * sin_thetas;
  const torch::Tensor sh_directions_z = cos_thetas;

  const std::vector<torch::Tensor> sh_directions_vec(
      {sh_directions_x, sh_directions_y, sh_directions_z});

  const torch::Tensor sh_directions =
      torch::vstack(sh_directions_vec).transpose(1, 0);

  return sh_directions;
}

const torch::Tensor toPolars(const torch::Tensor &directions) {
  const torch::Tensor phis = torch::atan2(directions.index({slice_all, 1}),
                                          directions.index({slice_all, 0}));

  const torch::Tensor thetas = torch::acos(directions.index({slice_all, 2}));

  const torch::Tensor polars = torch::vstack({phis, thetas}).transpose(1, 0);

  return polars;
}
