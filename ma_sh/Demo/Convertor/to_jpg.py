from ma_sh.Config.custom_path import toDatasetRootPath
from ma_sh.Module.Convertor.to_jpg import Convertor


def demo():
    dataset_root_folder_path = toDatasetRootPath()
    quality = 95
    source_data_type = '.png'
    target_data_type = '.jpg'

    if dataset_root_folder_path is None:
        print('[ERROR][to_jpg::demo]')
        print('\t toDatasetRootPath failed!')
        return False

    convertor = Convertor(
        dataset_root_folder_path + "Objaverse_82K/render/",
        dataset_root_folder_path + "Objaverse_82K/render_jpg/",
        quality,
    )

    convertor.convertAll(source_data_type, target_data_type)
    return True
