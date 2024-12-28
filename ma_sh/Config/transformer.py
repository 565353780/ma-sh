import os
from typing import Union

from ma_sh.Config.custom_path import toDatasetRootPath
from distribution_manage.Module.transformer import Transformer

VALID_TRANSFORMER_IDS = [
    'ShapeNet',
    'ShapeNet_03001627',
    'Objaverse_82K',
]


def getTransformer(transformer_id: str = 'Objaverse_82K') -> Union[None, Transformer]:
    if transformer_id not in VALID_TRANSFORMER_IDS:
        print('[ERROR][transformer::getTransformer]')
        print('\t transformer id not valid!')
        print('\t transformer_id:', transformer_id)
        print('\t valid transformer ids:', VALID_TRANSFORMER_IDS)
        return None

    dataset_root_folder_path = toDatasetRootPath()
    if dataset_root_folder_path is None:
        print('[ERROR][transformer::getTransformer]')
        print('\t toDatasetRootPath failed!')
        return None

    transformer_file_path = dataset_root_folder_path + 'Transformers/' + transformer_id + '.pkl'
    if not os.path.exists(transformer_file_path):
        print('[ERROR][transformer::getTransformer]')
        print('\t transformer file not exist!')
        return None

    transformer = Transformer(transformer_file_path)

    return transformer
