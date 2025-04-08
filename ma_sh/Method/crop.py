import os
import numpy as np
import open3d as o3d
from tqdm import trange
from typing import Union

from ma_sh.Method.path import removeFile
from ma_sh.Method.pcd import getPointCloud


def rotate_and_crop_point_cloud(
    points: np.ndarray,
    angle: float = 0.0,
    cut_weight: float = 0.0,
    is_crop_right: bool = True,
) -> np.ndarray:
    rad = angle * np.pi / 180.0

    cos_theta = np.cos(rad)
    sin_theta = np.sin(rad)
    Rz = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta,  cos_theta, 0],
        [0, 0, 1]
    ])

    rotated_points = points @ Rz.T

    x_vals = rotated_points[:, 0]
    x_min, x_max = x_vals.min(), x_vals.max()
    x_cutoff = x_min + cut_weight * (x_max - x_min)

    if is_crop_right:
        mask = x_vals >= x_cutoff
    else:
        mask = x_vals <= x_cutoff

    cropped_rotated_points = rotated_points[mask]

    cropped_points = cropped_rotated_points @ Rz

    return cropped_points

def createCroppedPcdFiles(
    points_data: Union[np.ndarray, str],
    save_pcd_folder_path: str,
    crop_num: int = 60,
    angle: float = 0.0,
    is_crop_right: bool = True,
    render: bool = False,
    overwrite: bool = False,
) -> bool:
    if isinstance(points_data, str):
        pcd_file_path = points_data

        if not os.path.exists(pcd_file_path):
            print('[ERROR][crop::createCroppedPcdFiles]')
            print('\t pcd file not found!')
            print('\t pcd_file_path:', pcd_file_path)
            return False

        if pcd_file_path.endswith('.npy'):
            points = np.load(pcd_file_path)
        else:
            pcd = o3d.io.read_point_cloud(pcd_file_path)
            points = np.asarray(pcd.points)

    os.makedirs(save_pcd_folder_path, exist_ok=True)

    print('[INFO][crop::createCroppedPcdFiles]')
    print('\t start create cropped pcd...')
    for i in trange(1, crop_num):
        save_pcd_file_path = save_pcd_folder_path + str(i) + '_pcd.ply'
        if os.path.exists(save_pcd_file_path):
            if not overwrite:
                continue

            removeFile(save_pcd_file_path)

        cut_weight = 1.0 * i / crop_num

        cropped_pts = rotate_and_crop_point_cloud(points, angle, cut_weight, is_crop_right)

        cropped_pcd = getPointCloud(cropped_pts)

        if render:
            o3d.visualization.draw_geometries([cropped_pcd])

        # createFileFolder(save_pcd_file_path)
        o3d.io.write_point_cloud(save_pcd_file_path, cropped_pcd, write_ascii=True)

    return True
