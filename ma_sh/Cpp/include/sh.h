#pragma once

#include <torch/extension.h>

const double toSHWeight(const int &degree, const int &real_idx);

const torch::Tensor toSHCommonValue(const torch::Tensor &phis,
                                    const torch::Tensor &thetas,
                                    const int &idx);

const torch::Tensor toDeg1ThetaValue(const torch::Tensor &thetas,
                                     const int &real_idx);

const torch::Tensor toDeg2ThetaValue(const torch::Tensor &thetas,
                                     const int &real_idx);

const torch::Tensor toDeg3ThetaValue(const torch::Tensor &thetas,
                                     const int &real_idx);

const torch::Tensor toDeg4ThetaValue(const torch::Tensor &thetas,
                                     const int &real_idx);

const torch::Tensor toDeg5ThetaValue(const torch::Tensor &thetas,
                                     const int &real_idx);

const torch::Tensor toDeg6ThetaValue(const torch::Tensor &thetas,
                                     const int &real_idx);

const torch::Tensor toSHResValue(const torch::Tensor &thetas, const int &degree,
                                 const int &real_idx);

const torch::Tensor toSHBaseValue(const torch::Tensor &phis,
                                  const torch::Tensor &thetas,
                                  const int &degree, const int &idx);

const torch::Tensor toSHBaseValues(const torch::Tensor &phis,
                                   const torch::Tensor &thetas,
                                   const int &degree_max);

const torch::Tensor toSHDirections(const torch::Tensor &phis,
                                   const torch::Tensor &thetas);
