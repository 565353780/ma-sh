from ma_sh.Config.custom_path import toDatasetRootPath
from ma_sh.Module.Convertor.images_to_video import Convertor


def demo():
    dataset_root_folder_path = toDatasetRootPath()
    video_width = 540
    video_height = 540
    video_fps = 24
    background_color = [255, 255, 255]
    overwrite = True
    source_data_type = '/'
    target_data_type = '.mp4'

    if dataset_root_folder_path is None:
        print('[ERROR][images_to_video::demo]')
        print('\t toDatasetRootPath failed!')
        return False

    convertor = Convertor(
        '/home/chli/chLi/Results/ma-sh/output/fit_render/',
        '/home/chli/chLi/Results/ma-sh/output/fit_video/',
        video_width,
        video_height,
        video_fps,
        background_color,
        overwrite)

    convertor.convertAll(source_data_type, target_data_type)
    return True
