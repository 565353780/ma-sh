import os


class PipelineConvertor(object):
    def __init__(
        self,
        convertor_list: list = [],
    ) -> None:
        self.convertor_list = convertor_list
        return

    def convertOneShape(
        self,
        rel_base_path: str,
        data_type_list: list,
    ) -> bool:
        if len(data_type_list) < 2 or len(data_type_list) - 1 != len(self.convertor_list):
            print('[ERROR][PipelineConvertor::convertOneShape]')
            print('\t data type num not valid!')
            return False

        for i in range(len(self.convertor_list)):
            source_data_type = data_type_list[i]
            target_data_type = data_type_list[i + 1]

            if not self.convertor_list[i].convertOneShape(rel_base_path, source_data_type, target_data_type):
                print('[ERROR][PipelineConvertor::convertOneShape]')
                print('\t convertOneShape failed for convertor[' + str(i) + ']!')
                return False

        return True

    def convertAll(self, data_type_list: list) -> bool:
        if len(self.convertor_list) == 0:
            return True

        if len(data_type_list) < 2 or len(data_type_list) - 1 != len(self.convertor_list):
            print('[ERROR][PipelineConvertor::convertAll]')
            print('\t data type num not valid!')
            return False

        print("[INFO][PipelineConvertor::convertAll]")
        print("\t start convert all data...")
        solved_shape_num = 0

        for root, dirs, files in os.walk(self.convertor_list[0].source_root_folder_path):
            if data_type_list[0] == '/':
                if dirs:
                    continue

                rel_base_path = os.path.relpath(root, self.convertor_list[0].source_root_folder_path)

                self.convertOneShape(rel_base_path, data_type_list)

                solved_shape_num += 1
                print("solved shape num:", solved_shape_num)
                continue

            for file in files:
                if not file.endswith(data_type_list[0]):
                    continue

                if file.endswith('_tmp' + data_type_list[1]):
                    continue

                rel_base_path = os.path.relpath(root + '/' + file, self.convertor_list[0].source_root_folder_path)

                rel_base_path = rel_base_path[:-len(data_type_list[0])]

                self.convertOneShape(rel_base_path, data_type_list)

                solved_shape_num += 1
                print("solved shape num:", solved_shape_num)

        return True
