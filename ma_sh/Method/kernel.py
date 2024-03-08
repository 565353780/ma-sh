from ma_sh.Config.kernel import (
    toUniformSamplePhisDict,
    toUniformSampleThetasDict,
    toBoundIdxsDict,
    toMaskBoundaryPhisDict,
    toMaskBaseValuesDict, 
    toMaskValuesDict,
    toMaskBoundaryMaxThetasDict,
    toInMaxMaskSamplePolarIdxsListDict,
    toInMaskSamplePolarCountsDict,
)

test_mode = 'c'

toUniformSamplePhis = toUniformSamplePhisDict[test_mode]
toUniformSampleThetas = toUniformSampleThetasDict[test_mode]
toBoundIdxs = toBoundIdxsDict[test_mode]
toMaskBoundaryPhis = toMaskBoundaryPhisDict[test_mode]
toMaskBaseValues = toMaskBaseValuesDict[test_mode]
toMaskValues = toMaskValuesDict[test_mode]
toMaskBoundaryMaxThetas = toMaskBoundaryMaxThetasDict[test_mode]
toInMaxMaskSamplePolarIdxsList = toInMaxMaskSamplePolarIdxsListDict[test_mode]
toInMaskSamplePolarCounts = toInMaskSamplePolarCountsDict[test_mode]

if False:
    toUniformSamplePhis = toUniformSamplePhisDict['c']
    toUniformSampleThetas = toUniformSampleThetasDict['c']
    toBoundIdxs = toBoundIdxsDict['c']
    toMaskBoundaryPhis = toMaskBoundaryPhisDict['p+']
    toMaskBaseValues = toMaskBaseValuesDict['p+']
    toMaskValues = toMaskValuesDict['c']
    toMaskBoundaryMaxThetas = toMaskBoundaryMaxThetasDict['c']
    toInMaxMaskSamplePolarIdxsList = toInMaxMaskSamplePolarIdxsListDict['c']
    toInMaskSamplePolarCounts = toInMaskSamplePolarCountsDict['c']
