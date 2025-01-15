import os

from ma_sh.Method.video import imagesToVideo
from ma_sh.Module.Convertor.base_convertor import BaseConvertor


class Convertor(BaseConvertor):
    def __init__(
        self,
        source_root_folder_path: str,
        target_root_folder_path: str,
        video_width: int = 540,
        video_height: int = 540,
        video_fps: int = 24,
        background_color: list = [255, 255, 255],
    ) -> None:
        super().__init__(source_root_folder_path, target_root_folder_path)

        self.video_width = video_width
        self.video_height = video_height
        self.video_fps = video_fps
        self.background_color = background_color
        return

    def convertData(self, source_path: str, target_path: str) -> bool:
        image_filename_list = os.listdir(source_path)

        valid_image_filename_list = []

        for image_filename in image_filename_list:
            if image_filename[-4:] not in ['.png', '.jpg']:
                continue

            valid_image_filename_list.append(image_filename)

        if len(valid_image_filename_list) == 0:
            return True

        valid_image_filename_list.sort(key=lambda x: int(x.split('_')[0]))

        image_file_path_list = [source_path + image_filename for image_filename in valid_image_filename_list]

        if not imagesToVideo(
            image_file_path_list,
            target_path,
            self.video_width,
            self.video_height,
            self.video_fps,
            self.background_color,
            False):
            print("[ERROR][Convertor::convertData]")
            print("\t imagesToVideo failed!")
            print("\t source_path:", source_path)
            return False

        return True
