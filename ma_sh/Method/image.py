import os
import cv2
import numpy as np
from typing import Union


def paintPNGBackGround(
        png_image: np.ndarray,
        background_color: Union[np.ndarray, list] = [255, 255, 255]
) -> np.ndarray:
    float_background_color = np.asarray(background_color, dtype=float)

    float_png_image = png_image.astype(float)

    if png_image.dtype == np.uint16:
        float_png_image /= 256.0

    painted_png_image = np.zeros_like(png_image, dtype=float)

    rgb = float_png_image[:, :, :3]
    alpha = float_png_image[:, :, 3, None]

    float_background_image = np.ones_like(rgb, dtype=float)
    float_background_image[:, :, 0] = float_background_color[0]
    float_background_image[:, :, 1] = float_background_color[1]
    float_background_image[:, :, 2] = float_background_color[2]


    alpha_weight = alpha / 255.0
    float_painted_png_image = alpha_weight * rgb + (1 - alpha_weight) * float_background_image

    painted_png_image = float_painted_png_image.astype(np.uint8)

    return painted_png_image

def loadPNGWithBackGround(
        png_image_file_path: str,
        background_color: Union[np.ndarray, list] = [255, 255, 255]
) -> Union[np.ndarray, None]:
    if not os.path.exists(png_image_file_path):
        print('[ERROR][image::loadPNGWithBackGround]')
        print('\t png image file not exist!')
        print('\t png_image_file_path:', png_image_file_path)

        return None

    png_image = cv2.imread(png_image_file_path, cv2.IMREAD_UNCHANGED)

    painted_png_image = paintPNGBackGround(png_image, background_color)

    return painted_png_image
