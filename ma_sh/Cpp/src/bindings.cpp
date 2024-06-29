#include "bound.h"
#include "chamfer.h"
#include "direction.h"
#include "filter.h"
#include "fps.h"
#include "idx.h"
#include "inv.h"
#include "loss.h"
#include "mash.h"
#include "mash_unit.h"
#include "mask.h"
#include "rotate.h"
#include "sample.h"
#include "sh.h"
#include "simple_mash.h"
#include "single_ops.h"
#include "value.h"

#include <pybind11/pybind11.h>

PYBIND11_MODULE(mash_cpp, m) {
  m.doc() = "pybind11 mash cpp plugin";

  m.def("toAnchorBounds", &toAnchorBounds, "bound.toAnchorBounds");

  m.def("toChamferDistance", &toChamferDistance, "chamfer.toChamferDistance");

  m.def("toDirections", &toDirections, "direction.toDirections");
  m.def("toPolars", &toPolars, "direction.toPolars");

  m.def("toMaxValues", &toMaxValues, "filter.toMaxValues");

  m.def("toSingleFPSPointIdxs", &toSingleFPSPointIdxs,
        "fps.toSingleFPSPointIdxs");
  m.def("toFPSPointIdxs", &toFPSPointIdxs, "fps.toFPSPointIdxs");

  m.def("toCounts", &toCounts, "idx.toCounts");
  m.def("toIdxs", &toIdxs, "idx.toIdxs");
  m.def("toDataIdxs", &toDataIdxs, "idx.toDataIdxs");
  m.def("toIdxCounts", &toIdxCounts, "idx.toIdxCounts");
  m.def("toLowerIdxsVec", &toLowerIdxsVec, "idx.toLowerIdxsVec");

  m.def("toInvPoints", &toInvPoints, "inv.toInvPoints");

  m.def("toAnchorFitLoss", &toAnchorFitLoss, "loss.toAnchorFitLoss");
  m.def("toAnchorCoverageLoss", &toAnchorCoverageLoss,
        "loss.toAnchorCoverageLoss");
  m.def("toChamferDistanceLoss", &toChamferDistanceLoss,
        "loss.toChamferDistanceLoss");
  m.def("toAnchorChamferDistanceLoss", &toAnchorChamferDistanceLoss,
        "loss.toAnchorChamferDistanceLoss");
  m.def("toBoundaryConnectLoss", &toBoundaryConnectLoss,
        "loss.toBoundaryConnectLoss");

  m.def("toInMaskSamplePolars", &toInMaskSamplePolars,
        "mash.toInMaskSamplePolars");
  m.def("toSamplePoints", &toSamplePoints, "mash.toSamplePoints");
  m.def("toWeightedSamplePoints", &toWeightedSamplePoints,
        "mash.toWeightedSamplePoints");
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
  m.def("toFPSPoints", &toFPSPoints, "mash_unit.toFPSPoints");

  m.def("toMaskBaseValues", &toMaskBaseValues, "mask.toMaskBaseValues");

  m.def("toRotateMatrixs", &toRotateMatrixs, "rotate.toRotateMatrixs");
  m.def("toRotateVectors", &toRotateVectors, "rotate.toRotateVectors");
  m.def("toRotateVectorsByFaceForwardVectors",
        &toRotateVectorsByFaceForwardVectors,
        "rotate.toRotateVectorsByFaceForwardVectors");

  m.def("toUniformSamplePhis", &toUniformSamplePhis,
        "sample.toUniformSamplePhis");
  m.def("toUniformSampleThetas", &toUniformSampleThetas,
        "sample.toUniformSampleThetas");
  m.def("toMaskBoundaryPhis", &toMaskBoundaryPhis, "sample.toMaskBoundaryPhis");
  m.def("toSampleThetaNums", &toSampleThetaNums, "sample.toSampleThetaNums");
  m.def("toSampleThetas", &toSampleThetas, "sample.toSampleThetas");

  m.def("toSimpleMashSamplePoints", &toSimpleMashSamplePoints,
        "simple_mash.toSimpleMashSamplePoints");

  m.def("toSHBaseValues", &toSHBaseValues, "sh.toSHBaseValues");

  m.def("toSingleRotateMatrix", &toSingleRotateMatrix,
        "single_ops.toSingleRotateMatrix");
  m.def("toSingleMaskBoundaryThetas", &toSingleMaskBoundaryThetas,
        "single_ops.toSingleMaskBoundaryThetas");
  m.def("toSingleSHDists", &toSingleSHDists, "single_ops.toSingleSHDists");

  m.def("toValues", &toValues, "value.toValues");
}
