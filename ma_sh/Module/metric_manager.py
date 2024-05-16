import numpy as np
from typing import Union

from ma_sh.Config.shapenet import SHAPENET_NAME_DICT

class MetricManager(object):
    def __init__(self, name: str) -> None:
        self.name = name
        self.values_dict = {}
        return

    def addValue(self, label: str, value: float) -> bool:
        if label not in self.values_dict.keys():
            self.values_dict[label] = []

        self.values_dict[label].append(value)
        return True

    def toMeanValue(self, label: str) -> Union[float, None]:
        if label not in self.values_dict.keys():
            print('[ERROR][MetricManager::toMeanValue]')
            print('\t label not exist!')
            print('\t label:', label)
            print('\t valid labels are:', self.values_dict.keys())
            return None

        values = self.values_dict[label]

        mean_value = np.mean(np.array(values, dtype=float)).item()
        return mean_value

    def outputInfo(self) -> bool:
        print('[' + self.name + ']')
        for key in self.values_dict.keys():
            if key in SHAPENET_NAME_DICT.keys():
                key_name = SHAPENET_NAME_DICT[key]
            else:
                key_name = key
                continue

            value = self.toMeanValue(key)
            if 'cd' in self.name:
                value *= 1000

            print('\t [' + key_name + ']', value)

        all_values = []
        for key, values in self.values_dict.items():
            if key not in SHAPENET_NAME_DICT.keys():
                continue

            all_values += values

        all_mean_value = np.mean(np.array(all_values, dtype=float)).item()
        if 'cd' in self.name:
            all_mean_value *= 1000
        print('\t [mean]', all_mean_value)
        return True
