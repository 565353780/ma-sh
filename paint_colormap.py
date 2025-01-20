import os
import numpy as np

from ma_sh.Data.mesh import Mesh
from ma_sh.Model.mash import Mash


def demo():
    error_max_percent = 0.0001
    accurate = False
    overwrite = False

    mesh_name = 'bunny'

    normalized_mesh_file_path = '/home/chli/chLi/Dataset/Famous/normalized_mesh/' + mesh_name + '.ply'
    gt_points_file_path = '/home/chli/chLi/Dataset/Famous/sample_pcd/' + mesh_name + '.npy'
    mash_result_folder_path = '/home/chli/chLi/Results/ma-sh/output/fit/fixed/' + mesh_name + '/'
    save_colored_gt_mesh_folder_path = '/home/chli/chLi/Results/ma-sh/output/fit_error_mesh/' + mesh_name + '/'
    save_colored_mash_folder_path = '/home/chli/chLi/Results/ma-sh/output/fit_error_mash/' + mesh_name + '/'

    assert os.path.exists(normalized_mesh_file_path)
    assert os.path.exists(gt_points_file_path)
    assert os.path.exists(mash_result_folder_path)

    gt_mesh = Mesh(normalized_mesh_file_path)

    for root, _, files in os.walk(mash_result_folder_path):
        for file in files:
            if not file.endswith('.npy'):
                continue

            rel_file_path = os.path.relpath(root, mash_result_folder_path) + '/'

            mash_file_path = root + '/' + file
            assert os.path.exists(mash_file_path)

            save_colored_gt_mesh_file_path = save_colored_gt_mesh_folder_path + rel_file_path + file[:-4] + '.ply'
            save_colored_mash_file_path = save_colored_mash_folder_path + rel_file_path + file[:-4] + '.ply'

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

if __name__ == "__main__":
    demo()
