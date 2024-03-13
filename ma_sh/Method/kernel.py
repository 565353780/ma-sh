from ma_sh.Config.kernel import (
    toMaxValuesDict,
    toCountsDict,
    toIdxsDict,
    toLowerIdxsListDict,
    toMaskBaseValuesDict,
    toRotateMatrixsDict,
    toRotateVectorsDict,
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

toRotateMatrixs = toRotateMatrixsDict[test_mode]
toRotateVectors = toRotateVectorsDict[test_mode]

toUniformSamplePhis = toUniformSamplePhisDict[test_mode]
toUniformSampleThetas = toUniformSampleThetasDict[test_mode]
toMaskBoundaryPhis = toMaskBoundaryPhisDict[test_mode]

toSHBaseValues = toSHBaseValuesDict[test_mode]

toValues = toValuesDict[test_mode]

if False:
    toMaxValues = toMaxValuesDict["c"]

    toCounts = toCountsDict["c"]
    toIdxs = toIdxsDict["c"]
    toLowerIdxsList = toLowerIdxsListDict["c"]

    toMaskBaseValues = toMaskBaseValuesDict["p+"]

    toRotateMatrixs = toRotateMatrixsDict['c']
    toRotateVectors = toRotateVectorsDict['c']

    toUniformSamplePhis = toUniformSamplePhisDict["c"]
    toUniformSampleThetas = toUniformSampleThetasDict["c"]
    toMaskBoundaryPhis = toMaskBoundaryPhisDict["p+"]

    toSHBaseValues = toSHBaseValuesDict["c"]

    toValues = toValuesDict["c"]
