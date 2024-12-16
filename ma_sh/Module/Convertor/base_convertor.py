import os
from pathlib import Path
from abc import ABC, abstractmethod

from ma_sh.Method.path import createFileFolder, removeFile, renameFile, renameFolder, isHaveSubFolder


class BaseConvertor(ABC):
    def __init__(
        self,
        source_root_folder_path: str,
        target_root_folder_path: str,
    ) -> None:
        self.source_root_folder_path = source_root_folder_path
        self.target_root_folder_path = target_root_folder_path
        return

    @abstractmethod
    def convertData(self, source_path: str, target_path: str) -> bool:
        pass

    def convertOneShape(
        self,
        rel_base_path: str,
        source_data_type: str,
        target_data_type: str,
    ) -> bool:
        target_path = (
            self.target_root_folder_path + rel_base_path + target_data_type
        )

        if os.path.exists(target_path):
            return True

        source_path = (
            self.source_root_folder_path + rel_base_path + source_data_type
        )

        if not os.path.exists(source_path):
            print("[ERROR][BaseConvertor::convertOneShape]")
            print("\t source file not exist!")
            print("\t source_path:", source_path)
            return False

        start_tag_file_path = (
            self.target_root_folder_path + rel_base_path + "_start.txt"
        )

        if os.path.exists(start_tag_file_path):
            return True

        createFileFolder(start_tag_file_path)

        with open(start_tag_file_path, "w") as f:
            f.write("\n")

        if target_data_type == '/':
            os.makedirs(target_path, exist_ok=True)
        else:
            createFileFolder(target_path)

        tmp_target_path = self.target_root_folder_path + rel_base_path + '_tmp' + target_data_type

        if not self.convertData(source_path, tmp_target_path):
            print('[ERROR][BaseConvertor::convertOneShape]')
            print('\t convertData failed!')
            return False

        if target_data_type == '/':
            renameFolder(tmp_target_path, target_path)
        else:
            renameFile(tmp_target_path, target_path)

        removeFile(start_tag_file_path)

        return True

    def convertAll(self, source_data_type: str, target_data_type: str) -> bool:
        print("[INFO][BaseConvertor::convertAll]")
        print("\t start convert all data...")
        solved_shape_num = 0

        for path in Path(self.source_root_folder_path).rglob("*"):
            if source_data_type == '/':
                if not path.is_dir():
                    continue

                if isHaveSubFolder(path):
                    continue

                rel_base_path = os.path.relpath(str(path), self.source_root_folder_path)

                self.convertOneShape(rel_base_path, source_data_type, target_data_type)

                solved_shape_num += 1
                print("solved shape num:", solved_shape_num)
                continue

            if not path.is_file():
                continue

            if not path.name.endswith(target_data_type):
                continue

            rel_base_path = os.path.relpath(str(path), self.source_root_folder_path)

            rel_base_path = rel_base_path[:-len(target_data_type)]

            self.convertOneShape(rel_base_path, source_data_type, target_data_type)

            solved_shape_num += 1
            print("solved shape num:", solved_shape_num)
        return True
