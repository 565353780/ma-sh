import torch
import mash_cpp
from torch import compile

from ma_sh.Config.mode import BACKEND, TORCH_COMPILE
from ma_sh.Method.kernel_unit import (
    toMaxValues,
    toCounts,
    toIdxs,
    toLowerIdxsList,
    toMaskBaseValues,
    toRotateMatrixs,
    toUniformSamplePhis,
    toUniformSampleThetas,
    toMaskBoundaryPhis,
    toSHBaseValues,
    toSHDirections,
    toValues,
)
from ma_sh.Method.MashPy import mash_unit
from ma_sh.Method.MashPy import mash

toParamsDict = {
    "c": mash_unit.toParams,
    "p": mash_unit.toParams,
    "p+": compile(mash_unit.toParams, mode=TORCH_COMPILE),
}

toPreLoadDatasDict = {
    "c": mash.toPreLoadDatas,
    "p": mash.toPreLoadDatas,
    "p+": compile(mash.toPreLoadDatas, mode=TORCH_COMPILE),
}

toMashSamplePointsDict = {
    "c": mash_cpp.toMashSamplePoints,
    "p": mash.toMashSamplePoints,
    "p+": compile(mash.toMashSamplePoints, mode=TORCH_COMPILE),
}

toParams = toParamsDict[BACKEND]
toPreLoadDatas = toPreLoadDatasDict[BACKEND]
toMashSamplePoints = toMashSamplePointsDict[BACKEND]
