#include "add.h"
#include "filter.h"
#include "idx.h"
#include "mask.h"
#include "sample.h"
#include <pybind11/pybind11.h>

PYBIND11_MODULE(mash_cpp, m) {
  m.doc() = "pybind11 mash cpp plugin";

  m.def("add", &add, "add.add");

  m.def("toCounts", &toCounts, "idx.toCounts");
  m.def("toIdxs", &toIdxs, "idx.toIdxs");
  m.def("toLowerIdxsVec", &toLowerIdxsVec, "idx.toLowerIdxsVec");

  m.def("toMaskBoundaryPhis", &toMaskBoundaryPhis, "toMaskBoundaryPhis");
  m.def("toMaskBaseValues", &toMaskBaseValues, "toMaskBaseValues");
  m.def("toMaskValues", &toMaskValues, "toMaskValues");
  m.def("toUniformSamplePhis", &toUniformSamplePhis, "toUniformSamplePhis");
  m.def("toUniformSampleThetas", &toUniformSampleThetas,
        "toUniformSampleThetas");
  m.def("toMaskBoundaryMaxThetas", &toMaskBoundaryMaxThetas,
        "toMaskBoundaryMaxThetas");
}
