import sys
sys.path.append('../distribution-manage/')

from ma_sh.Config.custom_path import toDatasetRootPath
from ma_sh.Method.mash_distribution import getMashDistribution

if __name__ == "__main__":
    dataset_root_folder_path = toDatasetRootPath()

    if dataset_root_folder_path is None:
        print('[ERROR][get_mash_distribution::__main__]')
        print('\t dataset not found!')
        exit()

    mash_folder_path = dataset_root_folder_path + 'MashV4/'

    getMashDistribution(mash_folder_path)
