import torch
import mash_cpp
from torch import compile

from ma_sh.Method import (
    idx,
    mask
)

compile_mode = 'max-autotune'

toBoundIdxsDict = {
    'c': mash_cpp.toBoundIdxs,
    'p': idx.toBoundIdxs,
    'p+': compile(idx.toBoundIdxs, mode=compile_mode),
}

getMaskBaseValuesDict = {
    'c': mash_cpp.getMaskBaseValues,
    'p': mask.getMaskBaseValues,
    'p+': compile(mask.getMaskBaseValues, mode=compile_mode),
}

getMaskValuesDict = {
    # 'c': mash_cpp.getMaskValues,
    'p': mask.getMaskValues,
    'p+': compile(mask.getMaskValues, mode=compile_mode),
}
