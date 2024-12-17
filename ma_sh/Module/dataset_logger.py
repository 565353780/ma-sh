import os
from time import sleep

from ma_sh.Module.logger import Logger


class DatasetLogger(object):
    def __init__(self,
                 dataset_root_folder_path: str,
                 log_folder_path: str) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path
        self.logger = Logger(log_folder_path)
        return

    def recordDatasetState(self, target_rel_folder_path_dict: dict) -> bool:
        for key, target_rel_folder_path in target_rel_folder_path_dict.items():
            target_folder_path = self.dataset_root_folder_path + target_rel_folder_path

            if not os.path.exists(target_folder_path):
                continue

            valid_data_num = 0
            for _, _, files in os.walk(target_folder_path):
                for file in files:
                    if file.endswith('_start.txt'):
                        continue
                    if '_tmp.' in file:
                        continue

                    valid_data_num += 1

            if self.logger.isValid():
                self.logger.addScalar('DataNum/' + key, valid_data_num)
        return True

    def autoRecordDatasetState(self, target_rel_folder_path_dict: dict, sleep_second: float = 1.0) -> bool:
        while True:
            self.recordDatasetState(target_rel_folder_path_dict)
            sleep(sleep_second)
