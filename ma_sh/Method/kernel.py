from ma_sh.Config.kernel import (
    toUniformSamplePhisDict,
    toUniformSampleThetasDict,
    toCountsDict,
    toIdxsDict,
    toLowerIdxsListDict,
    toMaskBoundaryPhisDict,
    toMaskBaseValuesDict,
    toMaskValuesDict,
    toMaskBoundaryMaxThetasDict,
)

test_mode = "c"

toUniformSamplePhis = toUniformSamplePhisDict[test_mode]
toUniformSampleThetas = toUniformSampleThetasDict[test_mode]
toCounts = toCountsDict[test_mode]
toIdxs = toIdxsDict[test_mode]
toLowerIdxsList = toLowerIdxsListDict[test_mode]
toMaskBoundaryPhis = toMaskBoundaryPhisDict[test_mode]
toMaskBaseValues = toMaskBaseValuesDict[test_mode]
toMaskValues = toMaskValuesDict[test_mode]
toMaskBoundaryMaxThetas = toMaskBoundaryMaxThetasDict[test_mode]

if False:
    toUniformSamplePhis = toUniformSamplePhisDict["c"]
    toUniformSampleThetas = toUniformSampleThetasDict["c"]
    toCounts = toCountsDict["c"]
    toIdxs = toIdxsDict["c"]
    toLowerIdxsList = toLowerIdxsListDict["c"]
    toMaskBoundaryPhis = toMaskBoundaryPhisDict["p+"]
    toMaskBaseValues = toMaskBaseValuesDict["p+"]
    toMaskValues = toMaskValuesDict["c"]
    toMaskBoundaryMaxThetas = toMaskBoundaryMaxThetasDict["c"]
