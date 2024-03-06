#include "add.h"
#include "idx.h"
#include "mask.h"
#include <pybind11/pybind11.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "pybind11 mash cpp plugin";
  m.def("add", &add, "A function which adds two numbers");
  m.def("toBoundIdxs", &toBoundIdxs, "toBoundIdxs");
  m.def("getMaskBaseValues", &getMaskBaseValues, "getMaskBaseValues");
}
