import os
import functools
import numpy as np
import open3d as o3d
from tqdm import tqdm

from ma_sh.Data.mesh import Mesh
from ma_sh.Method.path import createFileFolder
from ma_sh.Method.render import renderGeometries

def sortById(x: str, y: str) -> int:
    x_id = int(x.split('_')[0])
    y_id = int(y.split('_')[0])

    if x_id < y_id:
        return -1
    if x_id == y_id:
        return 0
    return 1

def demo():
    anchor_num = 800
    error_max_percent = 0.0001
    accurate = False
    overwrite = True

    save_colored_gt_mesh_folder_path = '/home/chli/Nutstore Files/MASH-Materials/teaser_materials/Anc-' + str(anchor_num) + '_ColoredGTMesh/'

    famous_filename_list = ['hand', 'dragon', 'xyzrgb_dragon_clean', 'angel', 'Armadillo', 'bunny', 'Liberty', 'xyzrgb_statuette']

    for mesh_name in famous_filename_list:
        mesh_file_path = '/home/chli/Dataset/Famous/' + mesh_name + '.ply'
        assert os.path.exists(mesh_file_path)

        normalized_mesh_file_path = '/home/chli/chLi/Dataset/NormalizedMesh/Famous/' + mesh_name + '.ply'
        if not os.path.exists(normalized_mesh_file_path):
            createFileFolder(normalized_mesh_file_path)
            mesh = Mesh(mesh_file_path)

            min_bound = np.min(mesh.vertices, axis=0)
            max_bound = np.max(mesh.vertices, axis=0)
            length = np.max(max_bound - min_bound)
            scale = 0.9 / length
            center = (min_bound + max_bound) / 2.0

            mesh.vertices = (mesh.vertices - center) * scale

            mesh.save(normalized_mesh_file_path, True)

        mash_pcd_folder_path = "/home/chli/Nutstore Files/MASH-Materials/teaser_materials/Anc-" + str(anchor_num) + '/' + mesh_name + '/pcd/'
        if not os.path.exists(mash_pcd_folder_path):
            continue

        mash_pcd_filename_list = os.listdir(mash_pcd_folder_path)
        mash_pcd_filename_list.sort(key=functools.cmp_to_key(sortById))

        if False:
            first_mash_pcd_file_path = mash_pcd_folder_path + mash_pcd_filename_list[0]

            first_gt_mesh = Mesh(normalized_mesh_file_path)

            first_mash_pcd = o3d.io.read_point_cloud(first_mash_pcd_file_path)

            first_mash_pts = np.asarray(first_mash_pcd.points)

            first_gt_mesh.paintJetColorsByPoints(first_mash_pts, error_max_percent, accurate)

            last_mash_pcd_file_path = mash_pcd_folder_path + mash_pcd_filename_list[-1]

            last_gt_mesh = Mesh(normalized_mesh_file_path)

            last_mash_pcd = o3d.io.read_point_cloud(last_mash_pcd_file_path)

            last_mash_pts = np.asarray(last_mash_pcd.points)

            last_gt_mesh.paintJetColorsByPoints(last_mash_pts, error_max_percent, accurate)

            first_mesh = first_gt_mesh.toO3DMesh()
            last_mesh = last_gt_mesh.toO3DMesh()
            last_mesh.translate([0, 1, 0])

            mesh_list = [first_mesh, last_mesh]

            print('start render mash:', mesh_name)
            renderGeometries(mesh_list)
            exit()

        print('start paint mesh:', mesh_name)
        for i, mash_pcd_filename in enumerate(tqdm(mash_pcd_filename_list)):
            if mash_pcd_filename[-4:] != '.ply':
                continue

            mash_pcd_id = int(mash_pcd_filename.split('_')[0])

            if mash_pcd_id not in [0, 20, 40, 60] or i == len(mash_pcd_filename_list) - 1:
                continue

            save_colored_gt_mesh_file_path = save_colored_gt_mesh_folder_path + mesh_name + '/' + mash_pcd_filename
            if os.path.exists(save_colored_gt_mesh_file_path):
                continue

            mash_pcd_file_path = mash_pcd_folder_path + mash_pcd_filename

            gt_mesh = Mesh(normalized_mesh_file_path)

            mash_pcd = o3d.io.read_point_cloud(mash_pcd_file_path)

            mash_pts = np.asarray(mash_pcd.points)

            gt_mesh.paintJetColorsByPoints(mash_pts, error_max_percent, accurate)

            gt_mesh.save(save_colored_gt_mesh_file_path, overwrite)
    return True

if __name__ == "__main__":
    demo()
