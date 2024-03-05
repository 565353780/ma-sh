#include <iostream>
#include <pybind11/pybind11.h>
#include <torch/torch.h>

namespace py = pybind11;

int add(int i, int j) { return i + j; }

PYBIND11_MODULE(pybind11_exp, m) {
  m.doc() = "pybind11 example plugin";
  m.def("add", &add, "A function which adds two numbers");
}
