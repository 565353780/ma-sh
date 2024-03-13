#pragma once

#include <torch/extension.h>

const torch::Tensor toRotateMatrixs(const torch::Tensor &rotate_vectors);

const torch::Tensor toRotateVectors(const torch::Tensor &rotate_matrixs);
