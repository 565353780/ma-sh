import os
import numpy as np

from ma_sh.Data.mesh import Mesh
from ma_sh.Model.mash import Mash


def paintColormap(
    normalized_mesh_file_path: str,
    mash_result_folder_path: str,
    save_colored_gt_mesh_folder_path: str,
    error_max_percent: float = 0.0001,
    accurate: bool = False,
    overwrite: bool = False,
):
    if not os.path.exists(normalized_mesh_file_path):
        print('[ERROR][paint::paintColormap]')
        print('\t normalized mesh not found!')
        print('\t normalized_mesh_file_path:', normalized_mesh_file_path)
        return False

    if not os.path.exists(mash_result_folder_path):
        print('[ERROR][paint::paintColormap]')
        print('\t mash result not found!')
        print('\t mash_result_folder_path:', mash_result_folder_path)
        return False

    gt_mesh = Mesh(normalized_mesh_file_path)

    for root, _, files in os.walk(mash_result_folder_path):
        for file in files:
            if not file.endswith('.npy'):
                continue

            rel_file_path = os.path.relpath(root, mash_result_folder_path) + '/'

            mash_file_path = root + '/' + file
            assert os.path.exists(mash_file_path)

            save_colored_gt_mesh_file_path = save_colored_gt_mesh_folder_path + rel_file_path + file[:-4] + '.ply'

            if not overwrite:
                if os.path.exists(save_colored_gt_mesh_file_path):
                    continue

            print('start paint mesh:', file)

            mash_pcd = Mash.fromParamsFile(
                mash_file_path,
                mask_boundary_sample_num=180,
                sample_polar_num=2000,
                sample_point_scale=1.0,
                device='cuda',
            ).toSamplePcd()

            mash_pts = np.asarray(mash_pcd.points)

            gt_mesh.paintJetColorsByPoints(mash_pts, error_max_percent, accurate)

            gt_mesh.save(save_colored_gt_mesh_file_path, overwrite)
    return True
