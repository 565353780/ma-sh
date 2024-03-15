from ma_sh.Config.mode import BACKEND
from ma_sh.Config.kernel import (
    toMaxValuesDict,
    toCountsDict,
    toIdxsDict,
    toDataIdxsDict,
    toLowerIdxsListDict,
    toMaskBaseValuesDict,
    toRotateMatrixsDict,
    toRotateVectorsDict,
    toUniformSamplePhisDict,
    toUniformSampleThetasDict,
    toMaskBoundaryPhisDict,
    toSHBaseValuesDict,
    toSHDirectionsDict,
    toValuesDict,
)


toMaxValues = toMaxValuesDict[BACKEND]

toCounts = toCountsDict[BACKEND]
toIdxs = toIdxsDict[BACKEND]
toDataIdxs = toDataIdxsDict[BACKEND]
toLowerIdxsList = toLowerIdxsListDict[BACKEND]

toMaskBaseValues = toMaskBaseValuesDict[BACKEND]

toRotateMatrixs = toRotateMatrixsDict[BACKEND]
toRotateVectors = toRotateVectorsDict[BACKEND]

toUniformSamplePhis = toUniformSamplePhisDict[BACKEND]
toUniformSampleThetas = toUniformSampleThetasDict[BACKEND]
# toMaskBoundaryPhis = toMaskBoundaryPhisDict["p+"]
toMaskBoundaryPhis = toMaskBoundaryPhisDict[BACKEND]

toSHBaseValues = toSHBaseValuesDict[BACKEND]
toSHDirections = toSHDirectionsDict[BACKEND]

toValues = toValuesDict[BACKEND]
