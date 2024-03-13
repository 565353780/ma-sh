import torch
import mash_cpp
from torch import compile

from ma_sh.Method import (
    filter,
    idx,
    mask,
    rotate,
    sample,
    sh,
    value,
)

compile_mode = "max-autotune"

toMaxValuesDict = {
    "c": mash_cpp.toMaxValues,
    "p": filter.toMaxValues,
    "p+": compile(filter.toMaxValues, mode=compile_mode),
}

toCountsDict = {
    "c": mash_cpp.toCounts,
    "p": idx.toCounts,
    "p+": compile(idx.toCounts, mode=compile_mode),
}

toIdxsDict = {
    "c": mash_cpp.toIdxs,
    "p": idx.toIdxs,
    "p+": compile(idx.toIdxs, mode=compile_mode),
}

toLowerIdxsListDict = {
    "c": mash_cpp.toLowerIdxsVec,
    "p": idx.toLowerIdxsList,
    "p+": compile(idx.toLowerIdxsList, mode=compile_mode),
}

toMaskBaseValuesDict = {
    "c": mash_cpp.toMaskBaseValues,
    "p": mask.toMaskBaseValues,
    "p+": compile(mask.toMaskBaseValues, mode=compile_mode),
}

toRotateMatrixsDict = {
    'c': mash_cpp.toRotateMatrixs,
    'p': rotate.toRotateMatrixs,
    "p+": compile(rotate.toRotateMatrixs, mode=compile_mode),
}

toRotateVectorsDict = {
    'c': mash_cpp.toRotateVectors,
    'p': rotate.toRotateVectors,
    "p+": compile(rotate.toRotateVectors, mode=compile_mode),
}

toUniformSamplePhisDict = {
    "c": mash_cpp.toUniformSamplePhis,
    "p": sample.toUniformSamplePhis,
    "p+": compile(sample.toUniformSamplePhis, mode=compile_mode),
}

toUniformSampleThetasDict = {
    "c": mash_cpp.toUniformSampleThetas,
    "p": sample.toUniformSampleThetas,
    "p+": compile(sample.toUniformSampleThetas, mode=compile_mode),
}

toMaskBoundaryPhisDict = {
    "c": mash_cpp.toMaskBoundaryPhis,
    "p": sample.toMaskBoundaryPhis,
    "p+": compile(sample.toMaskBoundaryPhis, mode=compile_mode),
}

toSHBaseValuesDict = {
    "c": mash_cpp.toSHBaseValues,
    "p": sh.toSHBaseValues,
    "p+": compile(sh.toSHBaseValues, mode=compile_mode),
}

toValuesDict = {
    "c": mash_cpp.toValues,
    "p": value.toValues,
    "p+": compile(value.toValues, mode=compile_mode),
}
