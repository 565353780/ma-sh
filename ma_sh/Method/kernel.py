from ma_sh.Config.kernel import (
    toMaxValuesDict,
    toCountsDict,
    toIdxsDict,
    toLowerIdxsListDict,
    toMaskBaseValuesDict,
    toUniformSamplePhisDict,
    toUniformSampleThetasDict,
    toMaskBoundaryPhisDict,
    toSHBaseValuesDict,
    toValuesDict,
)

test_mode = "c"

toMaxValues = toMaxValuesDict[test_mode]

toCounts = toCountsDict[test_mode]
toIdxs = toIdxsDict[test_mode]
toLowerIdxsList = toLowerIdxsListDict[test_mode]

toMaskBaseValues = toMaskBaseValuesDict[test_mode]

toUniformSamplePhis = toUniformSamplePhisDict[test_mode]
toUniformSampleThetas = toUniformSampleThetasDict[test_mode]
toMaskBoundaryPhis = toMaskBoundaryPhisDict[test_mode]

toSHBaseValues = toSHBaseValuesDict["c"]

toValues = toValuesDict[test_mode]

if False:
    toMaxValues = toMaxValuesDict["c"]

    toCounts = toCountsDict["c"]
    toIdxs = toIdxsDict["c"]
    toLowerIdxsList = toLowerIdxsListDict["c"]

    toMaskBaseValues = toMaskBaseValuesDict["p+"]

    toUniformSamplePhis = toUniformSamplePhisDict["c"]
    toUniformSampleThetas = toUniformSampleThetasDict["c"]
    toMaskBoundaryPhis = toMaskBoundaryPhisDict["p+"]

    toSHBaseValues = toSHBaseValuesDict["c"]

    toValues = toValuesDict["c"]
