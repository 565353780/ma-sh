import os
import cv2
import numpy as np
from tqdm import tqdm
from typing import Union

from ma_sh.Method.image import paintPNGBackGround
from ma_sh.Method.path import createFileFolder, removeFile, renameFile


def imagesToVideo(
    image_file_path_list: list,
    save_video_file_path: str,
    video_width: int = 540,
    video_height: int = 540,
    video_fps: int = 24,
    background_color: Union[np.ndarray, list] = [255, 255, 255],
    overwrite: bool = False
) -> bool:
    if os.path.exists(save_video_file_path):
        if not overwrite:
            return True

        removeFile(save_video_file_path)

    createFileFolder(save_video_file_path)

    tmp_save_video_file_path = save_video_file_path[:-4] + '_tmp' + save_video_file_path[-4:]

    video_size = (video_width, video_height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(tmp_save_video_file_path, fourcc, video_fps, video_size)

    print('[INFO][video::imagesToVideo]')
    print('\t start convert images to video...')
    for image_file_path in tqdm(image_file_path_list):
        if not os.path.exists(image_file_path):
            continue

        frame = cv2.imread(image_file_path, cv2.IMREAD_UNCHANGED)

        if image_file_path[-4:] == '.png':
            frame = paintPNGBackGround(frame, background_color)

        if (frame.shape[1], frame.shape[0]) != video_size:
            frame = cv2.resize(frame, video_size)

        video.write(frame)

    video.release()

    renameFile(tmp_save_video_file_path, save_video_file_path)

    return True
