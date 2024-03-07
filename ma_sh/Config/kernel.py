import torch
import mash_cpp
from torch import compile

from ma_sh.Method import (
    sample,
    idx,
    mask,
    filter,
)

compile_mode = 'max-autotune'

toUniformSamplePhisDict = {
    'c': mash_cpp.toUniformSamplePhis,
    'p': sample.toUniformSamplePhis,
    'p+': compile(sample.toUniformSamplePhis, mode=compile_mode),
}

toUniformSampleThetasDict = {
    'c': mash_cpp.toUniformSampleThetas,
    'p': sample.toUniformSampleThetas,
    'p+': compile(sample.toUniformSampleThetas, mode=compile_mode),
}

toBoundIdxsDict = {
    'c': mash_cpp.toBoundIdxs,
    'p': idx.toBoundIdxs,
    'p+': compile(idx.toBoundIdxs, mode=compile_mode),
}

toMaskBoundaryPhisDict = {
    'c': mash_cpp.toMaskBoundaryPhis,
    'p': sample.toMaskBoundaryPhis,
    'p+': compile(sample.toMaskBoundaryPhis, mode=compile_mode),
}

toMaskBaseValuesDict = {
    'c': mash_cpp.toMaskBaseValues,
    'p': mask.toMaskBaseValues,
    'p+': compile(mask.toMaskBaseValues, mode=compile_mode),
}

toMaskValuesDict = {
    'c': mash_cpp.toMaskValues,
    'p': mask.toMaskValues,
    'p+': compile(mask.toMaskValues, mode=compile_mode),
}

toMaskBoundaryMaxThetasDict = {
    'c': mash_cpp.toMaskBoundaryMaxThetas,
    'p': filter.toMaskBoundaryMaxThetas,
    'p+': compile(filter.toMaskBoundaryMaxThetas, mode=compile_mode),
}

toInMaskSamplePolarIdxsDict = {
    'c': mash_cpp.toInMaskSamplePolarIdxs,
    'p': idx.toInMaskSamplePolarIdxs,
    'p+': compile(idx.toInMaskSamplePolarIdxs, mode=compile_mode),
}
