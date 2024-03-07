from ma_sh.Config.kernel import (
    getUniformSamplePhisDict,
    getUniformSampleThetasDict,
    toBoundIdxsDict,
    toMaskBoundaryPhisDict,
    getMaskBaseValuesDict, 
    getMaskValuesDict,
)

test_mode = 'c'

getUniformSamplePhis = getUniformSamplePhisDict[test_mode]
getUniformSampleThetas = getUniformSampleThetasDict[test_mode]
toBoundIdxs = toBoundIdxsDict[test_mode]
toMaskBoundaryPhis = toMaskBoundaryPhisDict[test_mode]
getMaskBaseValues = getMaskBaseValuesDict[test_mode]
getMaskValues = getMaskValuesDict[test_mode]

if False:
    getUniformSamplePhis = getUniformSamplePhisDict['c']
    getUniformSampleThetas = getUniformSampleThetasDict['c']
    toBoundIdxs = toBoundIdxsDict['c']
    toMaskBoundaryPhis = toMaskBoundaryPhisDict['p+']
    getMaskBaseValues = getMaskBaseValuesDict['p+']
    getMaskValues = getMaskValuesDict['c']
