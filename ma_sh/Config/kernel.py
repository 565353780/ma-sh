import torch
import mash_cpp
from torch import compile

from ma_sh.Config.mode import TORCH_COMPILE
from ma_sh.Method.Mash import (
    filter,
    idx,
    mask,
    rotate,
    sample,
    sh,
    value,
)

toMaxValuesDict = {
    "c": mash_cpp.toMaxValues,
    "p": filter.toMaxValues,
    "p+": compile(filter.toMaxValues, mode=TORCH_COMPILE),
}

toCountsDict = {
    "c": mash_cpp.toCounts,
    "p": idx.toCounts,
    "p+": compile(idx.toCounts, mode=TORCH_COMPILE),
}

toIdxsDict = {
    "c": mash_cpp.toIdxs,
    "p": idx.toIdxs,
    "p+": compile(idx.toIdxs, mode=TORCH_COMPILE),
}

toDataIdxsDict = {
    "c": mash_cpp.toDataIdxs,
    "p": idx.toDataIdxs,
    "p+": compile(idx.toDataIdxs, mode=TORCH_COMPILE),
}

toLowerIdxsListDict = {
    "c": mash_cpp.toLowerIdxsVec,
    "p": idx.toLowerIdxsList,
    "p+": compile(idx.toLowerIdxsList, mode=TORCH_COMPILE),
}

toMaskBaseValuesDict = {
    "c": mash_cpp.toMaskBaseValues,
    "p": mask.toMaskBaseValues,
    "p+": compile(mask.toMaskBaseValues, mode=TORCH_COMPILE),
}

toRotateMatrixsDict = {
    "c": mash_cpp.toRotateMatrixs,
    "p": rotate.toRotateMatrixs,
    "p+": compile(rotate.toRotateMatrixs, mode=TORCH_COMPILE),
}

toRotateVectorsDict = {
    "c": mash_cpp.toRotateVectors,
    "p": rotate.toRotateVectors,
    "p+": compile(rotate.toRotateVectors, mode=TORCH_COMPILE),
}

toUniformSamplePhisDict = {
    "c": mash_cpp.toUniformSamplePhis,
    "p": sample.toUniformSamplePhis,
    "p+": compile(sample.toUniformSamplePhis, mode=TORCH_COMPILE),
}

toUniformSampleThetasDict = {
    "c": mash_cpp.toUniformSampleThetas,
    "p": sample.toUniformSampleThetas,
    "p+": compile(sample.toUniformSampleThetas, mode=TORCH_COMPILE),
}

toMaskBoundaryPhisDict = {
    "c": mash_cpp.toMaskBoundaryPhis,
    "p": sample.toMaskBoundaryPhis,
    "p+": compile(sample.toMaskBoundaryPhis, mode=TORCH_COMPILE),
}

toSHBaseValuesDict = {
    "c": mash_cpp.toSHBaseValues,
    "p": sh.toSHBaseValues,
    "p+": compile(sh.toSHBaseValues, mode=TORCH_COMPILE),
}

toSHDirectionsDict = {
    "c": mash_cpp.toSHDirections,
    "p": sh.toSHDirections,
    "p+": compile(sh.toSHDirections, mode=TORCH_COMPILE),
}

toValuesDict = {
    "c": mash_cpp.toValues,
    "p": value.toValues,
    "p+": compile(value.toValues, mode=TORCH_COMPILE),
}
