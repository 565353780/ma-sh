#include "ball_query.h"
#include "filter.h"
#include "group_points.h"
#include "idx.h"
#include "interpolate.h"
#include "inv.h"
#include "mash.h"
#include "mash_unit.h"
#include "mask.h"
#include "rotate.h"
#include "sample.h"
#include "sampling.h"
#include "sh.h"
#include "value.h"

#include <pybind11/pybind11.h>

PYBIND11_MODULE(mash_cpp, m) {
  m.doc() = "pybind11 mash cpp plugin";

  m.def("gather_points", &gather_points, "pointnet2_ops.gather_points");
  m.def("gather_points_grad", &gather_points_grad,
        "pointnet2_ops.gather_points_grad");
  m.def("furthest_point_sampling", &furthest_point_sampling,
        "pointnet2_ops.furthest_point_sampling");

  m.def("three_nn", &three_nn, "pointnet2_ops.three_nn");
  m.def("three_interpolate", &three_interpolate,
        "pointnet2_ops.three_interpolate");
  m.def("three_interpolate_grad", &three_interpolate_grad,
        "pointnet2_ops.three_interpolate_grad");

  m.def("ball_query", &ball_query, "pointnet2_ops.ball_query");

  m.def("group_points", &group_points, "pointnet2_ops.group_points");
  m.def("group_points_grad", &group_points_grad,
        "pointnet2_ops.group_points_grad");

  m.def("toMaxValues", &toMaxValues, "filter.toMaxValues");

  m.def("toCounts", &toCounts, "idx.toCounts");
  m.def("toIdxs", &toIdxs, "idx.toIdxs");
  m.def("toDataIdxs", &toDataIdxs, "idx.toDataIdxs");
  m.def("toLowerIdxsVec", &toLowerIdxsVec, "idx.toLowerIdxsVec");

  m.def("toInvPoints", &toInvPoints, "inv.toInvPoints");

  m.def("toMashSamplePoints", &toMashSamplePoints, "mash.toMashSamplePoints");

  m.def("toMaskBoundaryThetas", &toMaskBoundaryThetas,
        "mash_unit.toMaskBoundaryThetas");
  m.def("toInMaxMaskSamplePolarIdxsVec", &toInMaxMaskSamplePolarIdxsVec,
        "mash_unit.toInMaxMaskSamplePolarIdxsVec");
  m.def("toInMaxMaskSamplePolarIdxs", &toInMaxMaskSamplePolarIdxs,
        "mash_unit.toInMaxMaskSamplePolarIdxs");
  m.def("toInMaxMaskThetas", &toInMaxMaskThetas, "mash_unit.toInMaxMaskThetas");
  m.def("toInMaskSampleThetaWeights", &toInMaskSampleThetaWeights,
        "mash_unit.toInMaskSampleThetaWeights");
  m.def("toDetectThetas", &toDetectThetas, "mash_unit.toDetectThetas");
  m.def("toSHValues", &toSHValues, "mash_unit.toSHValues");
  m.def("toSHPoints", &toSHPoints, "mash_unit.toSHPoints");

  m.def("toMaskBaseValues", &toMaskBaseValues, "mask.toMaskBaseValues");

  m.def("toRotateMatrixs", &toRotateMatrixs, "rotate.toRotateMatrixs");
  m.def("toRotateVectors", &toRotateVectors, "rotate.toRotateVectors");

  m.def("toUniformSamplePhis", &toUniformSamplePhis,
        "sample.toUniformSamplePhis");
  m.def("toUniformSampleThetas", &toUniformSampleThetas,
        "sample.toUniformSampleThetas");
  m.def("toMaskBoundaryPhis", &toMaskBoundaryPhis, "sample.toMaskBoundaryPhis");
  m.def("toFPSPointIdxs", &toFPSPointIdxs, "sample.toFPSPointIdxs");

  m.def("toSHBaseValues", &toSHBaseValues, "sh.toSHBaseValues");
  m.def("toSHDirections", &toSHDirections, "sh.toSHDirections");

  m.def("toValues", &toValues, "value.toValues");
}
