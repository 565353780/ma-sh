#include "sh.h"
#include "ATen/core/ATen_fwd.h"
#include "weights.h"

const double toSHWeight(const int &degree, const int &real_idx) {
  switch (degree) {
  case 0: {
    return W0[real_idx];
  }
  case 1: {
    return W1[real_idx];
  }
  case 2: {
    return W2[real_idx];
  }
  case 3: {
    return W3[real_idx];
  }
  case 4: {
    return W4[real_idx];
  }
  case 5: {
    return W5[real_idx];
  }
  case 6: {
    return W6[real_idx];
  }
  default: {
    return 0.0;
  }
  }
}

const torch::Tensor toSHCommonValue(const torch::Tensor &phis,
                                    const torch::Tensor &thetas,
                                    const int &idx) {
  if (idx == 0) {
    const torch::TensorOptions opts =
        torch::TensorOptions().dtype(phis.dtype()).device(phis.device());

    return torch::tensor(1.0, opts);
  } else {
    const torch::Tensor sin_thetas = torch::sin(thetas);

    const torch::Tensor pow_sin_thetas = torch::pow(sin_thetas, std::abs(idx));

    if (idx > 0) {
      const torch::Tensor cos_phis = torch::cos(1.0 * idx * phis);

      const torch::Tensor common_value = cos_phis * pow_sin_thetas;

      return common_value;
    } else {

      const torch::Tensor sin_phis = torch::sin(-1.0 * idx * phis);

      const torch::Tensor common_value = sin_phis * pow_sin_thetas;

      return common_value;
    }
  }
}

const torch::Tensor toDeg1ThetaValue(const torch::Tensor &thetas,
                                     const int &real_idx) {
  switch (real_idx) {
  case 0: {
    return torch::cos(thetas);
  }
  default: {
    const torch::TensorOptions opts =
        torch::TensorOptions().dtype(thetas.dtype()).device(thetas.device());

    return torch::tensor(1.0, opts);
  }
  }
}

const torch::Tensor toDeg2ThetaValue(const torch::Tensor &thetas,
                                     const int &real_idx) {
  switch (real_idx) {
  case 0: {
    const torch::Tensor ct = torch::cos(thetas);

    return 3.0 * ct * ct - 1.0;
  }
  case 1: {
    return torch::cos(thetas);
  }
  default: {
    const torch::TensorOptions opts =
        torch::TensorOptions().dtype(thetas.dtype()).device(thetas.device());

    return torch::tensor(1.0, opts);
  }
  }
}

const torch::Tensor toDeg3ThetaValue(const torch::Tensor &thetas,
                                     const int &real_idx) {
  switch (real_idx) {
  case 0: {
    const torch::Tensor ct = torch::cos(thetas);

    return (5.0 * ct * ct - 3.0) * ct;
  }
  case 1: {
    const torch::Tensor ct = torch::cos(thetas);

    return 5.0 * ct * ct - 1.0;
  }
  case 2: {
    return torch::cos(thetas);
  }
  default: {
    const torch::TensorOptions opts =
        torch::TensorOptions().dtype(thetas.dtype()).device(thetas.device());

    return torch::tensor(1.0, opts);
  }
  }
}

const torch::Tensor toDeg4ThetaValue(const torch::Tensor &thetas,
                                     const int &real_idx) {
  switch (real_idx) {
  case 0: {
    const torch::Tensor ct = torch::cos(thetas);

    return (35.0 * ct * ct - 30.0) * ct * ct + 3.0;
  }
  case 1: {
    const torch::Tensor ct = torch::cos(thetas);

    return (7.0 * ct * ct - 3.0) * ct;
  }
  case 2: {
    const torch::Tensor ct = torch::cos(thetas);

    return 7.0 * ct * ct - 1.0;
  }
  case 3: {
    return torch::cos(thetas);
  }
  default: {
    const torch::TensorOptions opts =
        torch::TensorOptions().dtype(thetas.dtype()).device(thetas.device());

    return torch::tensor(1.0, opts);
  }
  }
}

const torch::Tensor toDeg5ThetaValue(const torch::Tensor &thetas,
                                     const int &real_idx) {
  switch (real_idx) {
  case 0: {
    const torch::Tensor ct = torch::cos(thetas);

    return ((63.0 * ct * ct - 70.0) * ct * ct + 15.0) * ct;
  }
  case 1: {
    const torch::Tensor ct = torch::cos(thetas);

    return (21.0 * ct * ct - 14.0) * ct * ct + 1.0;
  }
  case 2: {

    const torch::Tensor ct = torch::cos(thetas);

    return (3.0 * ct * ct - 1.0) * ct;
  }
  case 3: {
    const torch::Tensor ct = torch::cos(thetas);

    return 9.0 * ct * ct - 1.0;
  }
  case 4: {
    return torch::cos(thetas);
  }
  default: {
    const torch::TensorOptions opts =
        torch::TensorOptions().dtype(thetas.dtype()).device(thetas.device());

    return torch::tensor(1.0, opts);
  }
  }
}

const torch::Tensor toDeg6ThetaValue(const torch::Tensor &thetas,
                                     const int &real_idx) {
  switch (real_idx) {
  case 0: {
    const torch::Tensor ct = torch::cos(thetas);

    return ((231.0 * ct * ct - 315.0) * ct * ct + 105.0) * ct * ct - 5.0;
  }
  case 1: {
    const torch::Tensor ct = torch::cos(thetas);

    return ((33.0 * ct * ct - 30.0) * ct * ct + 5.0) * ct;
  }
  case 2: {
    const torch::Tensor ct = torch::cos(thetas);

    return (33.0 * ct * ct - 18.0) * ct * ct + 1.0;
  }
  case 3: {
    const torch::Tensor ct = torch::cos(thetas);

    return (11.0 * ct * ct - 3.0) * ct;
  }
  case 4: {
    const torch::Tensor ct = torch::cos(thetas);

    return 11.0 * ct * ct - 1.0;
  }
  case 5: {
    return torch::cos(thetas);
  }
  default: {
    const torch::TensorOptions opts =
        torch::TensorOptions().dtype(thetas.dtype()).device(thetas.device());

    return torch::tensor(1.0, opts);
  }
  }
}

const torch::Tensor toSHResValue(const torch::Tensor &thetas, const int &degree,
                                 const int &real_idx) {
  switch (degree) {
  case 0: {
    return torch::ones_like(thetas);
  }
  case 1: {
    return toDeg1ThetaValue(thetas, real_idx);
  }
  case 2: {
    return toDeg2ThetaValue(thetas, real_idx);
  }
  case 3: {
    return toDeg3ThetaValue(thetas, real_idx);
  }
  case 4: {
    return toDeg4ThetaValue(thetas, real_idx);
  }
  case 5: {
    return toDeg5ThetaValue(thetas, real_idx);
  }
  case 6: {
    return toDeg6ThetaValue(thetas, real_idx);
  }
  default: {
    const torch::TensorOptions opts =
        torch::TensorOptions().dtype(thetas.dtype()).device(thetas.device());

    return torch::tensor(0.0, opts);
  }
  }
}

const torch::Tensor toSHBaseValue(const torch::Tensor &phis,
                                  const torch::Tensor &thetas,
                                  const int &degree, const int &idx) {
  const int real_idx = std::abs(idx);

  const double sh_weight = toSHWeight(degree, real_idx);

  const torch::Tensor sh_common_value = toSHCommonValue(phis, thetas, idx);

  const torch::Tensor sh_res_value = toSHResValue(thetas, degree, real_idx);

  const torch::Tensor base_value = sh_weight * sh_common_value * sh_res_value;

  return base_value;
}

const torch::Tensor toSHBaseValues(const torch::Tensor &phis,
                                   const torch::Tensor &thetas,
                                   const int &degree_max) {
  std::vector<torch::Tensor> base_values_vec;
  base_values_vec.reserve((degree_max + 1) * (degree_max + 1));

  for (int degree = 0; degree < degree_max + 1; ++degree) {
    for (int idx = -degree; idx < degree + 1; ++idx) {
      const torch::Tensor current_base_value =
          toSHBaseValue(phis, thetas, degree, idx);

      base_values_vec.emplace_back(current_base_value);
    }
  }

  const torch::Tensor base_values = torch::vstack(base_values_vec);

  return base_values;
}

const torch::Tensor toSHDirections(const torch::Tensor &phis,
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
