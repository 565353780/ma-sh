#include "add.h"
#include "filter.h"
#include "idx.h"
#include "mask.h"
#include "sample.h"
#include <pybind11/pybind11.h>

PYBIND11_MODULE(mash_cpp, m) {
  m.doc() = "pybind11 mash cpp plugin";
  m.def("add", &add, "A function which adds two numbers");
  m.def("toBoundIdxs", &toBoundIdxs, "toBoundIdxs");
  m.def("toMaskBoundaryPhis", &toMaskBoundaryPhis, "toMaskBoundaryPhis");
  m.def("toMaskBaseValues", &toMaskBaseValues, "toMaskBaseValues");
  m.def("toMaskValues", &toMaskValues, "toMaskValues");
  m.def("toUniformSamplePhis", &toUniformSamplePhis, "toUniformSamplePhis");
  m.def("toUniformSampleThetas", &toUniformSampleThetas,
        "toUniformSampleThetas");
  m.def("toMaskBoundaryMaxThetas", &toMaskBoundaryMaxThetas,
        "toMaskBoundaryMaxThetas");
  m.def("toInMaskSamplePolarIdxs", &toInMaskSamplePolarIdxs,
        "toInMaskSamplePolarIdxs");
}
