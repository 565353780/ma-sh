import torch
import mash_cpp
from torch import compile

from ma_sh.Method import (
    sample,
    idx,
    mask,
)

compile_mode = 'max-autotune'

getUniformSamplePhisDict = {
    'c': mash_cpp.getUniformSamplePhis,
    'p': sample.getUniformSamplePhis,
    'p+': compile(sample.getUniformSamplePhis, mode=compile_mode),
}

getUniformSampleThetasDict = {
    'c': mash_cpp.getUniformSampleThetas,
    'p': sample.getUniformSampleThetas,
    'p+': compile(sample.getUniformSampleThetas, mode=compile_mode),
}

toBoundIdxsDict = {
    'c': mash_cpp.toBoundIdxs,
    'p': idx.toBoundIdxs,
    'p+': compile(idx.toBoundIdxs, mode=compile_mode),
}

toMaskBoundaryPhisDict = {
    'c': mash_cpp.toMaskBoundaryPhis,
    'p': idx.toMaskBoundaryPhis,
    'p+': compile(idx.toMaskBoundaryPhis, mode=compile_mode),
}

getMaskBaseValuesDict = {
    'c': mash_cpp.getMaskBaseValues,
    'p': mask.getMaskBaseValues,
    'p+': compile(mask.getMaskBaseValues, mode=compile_mode),
}

getMaskValuesDict = {
    'c': mash_cpp.getMaskValues,
    'p': mask.getMaskValues,
    'p+': compile(mask.getMaskValues, mode=compile_mode),
}
