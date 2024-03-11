from ma_sh.Config.kernel import (
    toUniformSamplePhisDict,
    toUniformSampleThetasDict,
    toMaskBoundaryPhisDict,
    toCountsDict,
    toIdxsDict,
    toLowerIdxsListDict,
    toMaxValuesDict,
    toMaskBaseValuesDict,
    toValuesDict,
)

test_mode = "c"

toUniformSamplePhis = toUniformSamplePhisDict[test_mode]
toUniformSampleThetas = toUniformSampleThetasDict[test_mode]
toMaskBoundaryPhis = toMaskBoundaryPhisDict[test_mode]

toCounts = toCountsDict[test_mode]
toIdxs = toIdxsDict[test_mode]
toLowerIdxsList = toLowerIdxsListDict[test_mode]
toMaxValues = toMaxValuesDict[test_mode]

toMaskBaseValues = toMaskBaseValuesDict[test_mode]
toValues = toValuesDict[test_mode]

if False:
    toUniformSamplePhis = toUniformSamplePhisDict["c"]
    toUniformSampleThetas = toUniformSampleThetasDict["c"]
    toMaskBoundaryPhis = toMaskBoundaryPhisDict["p+"]

    toCounts = toCountsDict["c"]
    toIdxs = toIdxsDict["c"]
    toLowerIdxsList = toLowerIdxsListDict["c"]
    toMaxValues = toMaxValuesDict["c"]

    toMaskBaseValues = toMaskBaseValuesDict["p+"]

    toValues = toValuesDict["c"]
