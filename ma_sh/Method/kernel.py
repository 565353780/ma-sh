from ma_sh.Config.kernel import (
    toUniformSamplePhisDict,
    toUniformSampleThetasDict,
    toBoundIdxsDict,
    toMaskBoundaryPhisDict,
    toMaskBaseValuesDict, 
    toMaskValuesDict,
    toMaskBoundaryMaxThetasDict,
    toInMaskSamplePolarIdxsDict,
)

test_mode = 'c'

toUniformSamplePhis = toUniformSamplePhisDict[test_mode]
toUniformSampleThetas = toUniformSampleThetasDict[test_mode]
toBoundIdxs = toBoundIdxsDict[test_mode]
toMaskBoundaryPhis = toMaskBoundaryPhisDict[test_mode]
toMaskBaseValues = toMaskBaseValuesDict[test_mode]
toMaskValues = toMaskValuesDict[test_mode]
toMaskBoundaryMaxThetas = toMaskBoundaryMaxThetasDict[test_mode]
toInMaskSamplePolarIdxs = toInMaskSamplePolarIdxsDict['c']

if False:
    toUniformSamplePhis = toUniformSamplePhisDict['c']
    toUniformSampleThetas = toUniformSampleThetasDict['c']
    toBoundIdxs = toBoundIdxsDict['c']
    toMaskBoundaryPhis = toMaskBoundaryPhisDict['p+']
    toMaskBaseValues = toMaskBaseValuesDict['p+']
    toMaskValues = toMaskValuesDict['c']
    toMaskBoundaryMaxThetas = toMaskBoundaryMaxThetasDict['c']
    toInMaskSamplePolarIdxs = toInMaskSamplePolarIdxsDict['p']
