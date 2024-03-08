from ma_sh.Config.kernel import (
    toUniformSamplePhisDict,
    toUniformSampleThetasDict,
    toCountsDict,
    toBoundIdxsDict,
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
toBoundIdxs = toBoundIdxsDict[test_mode]
toLowerIdxsList = toLowerIdxsListDict[test_mode]
toMaskBoundaryPhis = toMaskBoundaryPhisDict[test_mode]
toMaskBaseValues = toMaskBaseValuesDict[test_mode]
toMaskValues = toMaskValuesDict[test_mode]
toMaskBoundaryMaxThetas = toMaskBoundaryMaxThetasDict[test_mode]

if False:
    toUniformSamplePhis = toUniformSamplePhisDict["c"]
    toUniformSampleThetas = toUniformSampleThetasDict["c"]
    toCounts = toCountsDict["c"]
    toBoundIdxs = toBoundIdxsDict["c"]
    toLowerIdxsList = toLowerIdxsListDict["c"]
    toMaskBoundaryPhis = toMaskBoundaryPhisDict["p+"]
    toMaskBaseValues = toMaskBaseValuesDict["p+"]
    toMaskValues = toMaskValuesDict["c"]
    toMaskBoundaryMaxThetas = toMaskBoundaryMaxThetasDict["c"]
