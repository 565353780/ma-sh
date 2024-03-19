#pragma once

#include <torch/extension.h>

const torch::Tensor toRotateMatrixs(const torch::Tensor &rotate_vectors);

const torch::Tensor
toRotateVectorsByFaceForwardVectors(const torch::Tensor face_forward_vectors);

const torch::Tensor toRotateVectors(const torch::Tensor &rotate_matrixs);
