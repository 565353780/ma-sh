import torch
import mash_cpp
from torch import compile

from ma_sh.Method import (
    idx,
    mask
)

toBoundIdxsDict = {
    'c': mash_cpp.toBoundIdxs,
    'p': idx.toBoundIdxs,
    'p+': compile(idx.toBoundIdxs),
}

getMaskBaseValuesDict = {
    'c': mash_cpp.getMaskBaseValues,
    'p': mask.getMaskBaseValues,
    'p+': compile(mask.getMaskBaseValues),
}
