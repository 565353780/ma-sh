from ma_sh.Config.kernel import (
    toBoundIdxsDict,
    getMaskBaseValuesDict, 
    getMaskValuesDict,
)

toBoundIdxs = toBoundIdxsDict['c']
getMaskBaseValues = getMaskBaseValuesDict['c']
getMaskValues = getMaskValuesDict['c']

if False:
    toBoundIdxs = toBoundIdxsDict['c']
    getMaskBaseValues = getMaskBaseValuesDict['p+']
    getMaskValues = getMaskValuesDict['c']
