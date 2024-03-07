#include "add.h"
#include "idx.h"
#include "mask.h"
#include "sample.h"
#include <pybind11/pybind11.h>

PYBIND11_MODULE(mash_cpp, m) {
  m.doc() = "pybind11 mash cpp plugin";
  m.def("add", &add, "A function which adds two numbers");
  m.def("toBoundIdxs", &toBoundIdxs, "toBoundIdxs");
  m.def("getMaskBaseValues", &getMaskBaseValues, "getMaskBaseValues");
  m.def("getMaskValues", &getMaskValues, "getMaskValues");
  m.def("getUniformSamplePhis", &getUniformSamplePhis, "getUniformSamplePhis");
  m.def("getUniformSampleThetas", &getUniformSampleThetas,
        "getUniformSampleThetas");
}
