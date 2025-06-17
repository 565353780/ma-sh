#include "rotate.h"
#include "sh.h"

#include <pybind11/pybind11.h>

PYBIND11_MODULE(mash_cpp, m) {
  m.doc() = "pybind11 mash cpp plugin";

  m.def("toRotateMatrixs", &toRotateMatrixs, "rotate.toRotateMatrixs");
  m.def("toRotateVectors", &toRotateVectors, "rotate.toRotateVectors");
  m.def("toRotateVectorsByFaceForwardVectors",
        &toRotateVectorsByFaceForwardVectors,
        "rotate.toRotateVectorsByFaceForwardVectors");

  m.def("toSHBaseValues", &toSHBaseValues, "sh.toSHBaseValues");
}
