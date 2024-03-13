#include "add.h"
#include "filter.h"
#include "idx.h"
#include "mask.h"
#include "rotate.h"
#include "sample.h"
#include "sh.h"
#include "value.h"
#include <pybind11/pybind11.h>

PYBIND11_MODULE(mash_cpp, m) {
  m.doc() = "pybind11 mash cpp plugin";

  m.def("add", &add, "add.add");

  m.def("toMaxValues", &toMaxValues, "filter.toMaxValues");

  m.def("toCounts", &toCounts, "idx.toCounts");
  m.def("toIdxs", &toIdxs, "idx.toIdxs");
  m.def("toLowerIdxsVec", &toLowerIdxsVec, "idx.toLowerIdxsVec");

  m.def("toMaskBaseValues", &toMaskBaseValues, "mask.toMaskBaseValues");

  m.def("toRotateMatrixs", &toRotateMatrixs, "rotate.toRotateMatrixs");
  m.def("toRotateVectors", &toRotateVectors, "rotate.toRotateVectors");

  m.def("toUniformSamplePhis", &toUniformSamplePhis,
        "sample.toUniformSamplePhis");
  m.def("toUniformSampleThetas", &toUniformSampleThetas,
        "sample.toUniformSampleThetas");
  m.def("toMaskBoundaryPhis", &toMaskBoundaryPhis, "sample.toMaskBoundaryPhis");

  m.def("toSHBaseValues", &toSHBaseValues, "sh.toSHBaseValues");

  m.def("toValues", &toValues, "value.toValues");
}
