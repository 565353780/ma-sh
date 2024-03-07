from ma_sh.Config.kernel import (
    getUniformSamplePhisDict,
    getUniformSampleThetasDict,
    toBoundIdxsDict,
    getMaskBaseValuesDict, 
    getMaskValuesDict,
)

test_mode = 'p'

getUniformSamplePhis = getUniformSamplePhisDict[test_mode]
getUniformSampleThetas = getUniformSampleThetasDict[test_mode]
toBoundIdxs = toBoundIdxsDict[test_mode]
getMaskBaseValues = getMaskBaseValuesDict[test_mode]
getMaskValues = getMaskValuesDict[test_mode]

if False:
    getUniformSamplePhis = getUniformSamplePhisDict['c']
    getUniformSampleThetas = getUniformSampleThetasDict['c']
    toBoundIdxs = toBoundIdxsDict['c']
    getMaskBaseValues = getMaskBaseValuesDict['p+']
    getMaskValues = getMaskValuesDict['c']
