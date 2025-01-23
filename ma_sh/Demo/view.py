import os
import torch
from random import shuffle
import functools
import numpy as np
import open3d as o3d

from ma_sh.Config.custom_path import toDatasetRootPath
from ma_sh.Method.data import toNumpy
from ma_sh.Method.pcd import getPointCloud
from ma_sh.Method.clip import clip_with_obb
from ma_sh.Model.mash import Mash
from ma_sh.Module.o3d_viewer import O3DViewer


def compare(str_a: str, str_b: str) -> int:
    id_a = int(str_a.split("_")[0])
    id_b = int(str_b.split("_")[0])

    if id_a > id_b:
        return 1
    elif id_a == id_b:
        return 0
    return -1


def demo():
    # test clip function
    if False:
        pcd_file_path = '/home/chli/chLi/Dataset/Washer/sample_pcd/BOSCH_WLG.npy'
        pts = np.load(pcd_file_path)
        pcd = getPointCloud(pts)

        bounds = np.max(pts, axis=0) - np.min(pts, axis=0)

        clip_num = 6
        save_folder_path = '/home/chli/chLi/Results/ma-sh/output/clip/Washer/anc-1500/'
        for i in range(clip_num):
            current_center = 0.0 + i / (2 * clip_num + 2) # XiaomiSU7
            clipped_pts = clip_with_obb(pts, bounds, 0, [current_center, 0.0, 0.0])

            clipped_pcd = getPointCloud(clipped_pts)

            os.makedirs(save_folder_path, exist_ok=True)
            o3d.io.write_point_cloud(save_folder_path + str(i) + '_pcd.ply', clipped_pcd, write_ascii=True)
            #continue

            clipped_pcd.translate([1, 0, 0])

            print("start show mash:", pcd_file_path)
            o3d.visualization.draw_geometries([pcd, clipped_pcd])
        exit()

    # view single mash
    if False:
        mash_file_path_list = [
            '/home/chli/chLi/Results/ma-sh/output/fit/fixed/Washer/anchor-1500/mash/0_final_anc-1500_mash.npy',
            #'/home/chli/chLi/Results/ma-sh/output/fit/fixed/XiaomiSU7/anchor-1500/mash/0_final_anc-1500_mash.npy',
            #"/home/chli/chLi/Dataset/MashV4/ShapeNet/03001627/1006be65e7bc937e9141f9b58470d646.npy",
            #"/home/chli/chLi/Dataset/MashV4/ShapeNet/03001627/1007e20d5e811b308351982a6e40cf41.npy",
            #"./output/combined_mash.npy",
        ]

        for mash_file_path in mash_file_path_list:
            mash = Mash.fromParamsFile(mash_file_path, 180, 2000, 1.0, device="cuda")

            mash_pcd = mash.toSamplePcd()

            '''
            print("start show mash:", mash_file_path)
            o3d.visualization.draw_geometries([mash_pcd])
            exit()
            '''

            mash_pcd_pts = np.asarray(mash_pcd.points)

            bounds = np.max(mash_pcd_pts, axis=0) - np.min(mash_pcd_pts, axis=0)

            clip_num = 6
            save_folder_path = '/home/chli/chLi/Results/ma-sh/output/clip/XiaomiSU7/anc-1500/'
            for i in range(clip_num):
                current_center = 0.0 + i / (2 * clip_num + 2) # XiaomiSU7
                #current_center = 0.0 - i / (clip_num + 2) # Washer
                clipped_pts = clip_with_obb(mash_pcd_pts, bounds, 0, [current_center, 0.0, 0.0])

                clipped_pcd = getPointCloud(clipped_pts)

                os.makedirs(save_folder_path, exist_ok=True)
                o3d.io.write_point_cloud(save_folder_path + str(i) + '_pcd.ply', clipped_pcd, write_ascii=True)
                continue

                clipped_pcd.translate([1, 0, 0])

                print("start show mash:", mash_file_path)
                o3d.visualization.draw_geometries([mash_pcd, clipped_pcd])
        exit()

    # view mash dataset
    if False:
        dataset_root_folder_path = toDatasetRootPath()
        assert dataset_root_folder_path is not None

        # mash_dataset_folder_path = dataset_root_folder_path + 'Objaverse_82K/manifold_mash/'
        #'02828884', # 6: bench
        #'04256520', # 47: sofa
        mash_dataset_folder_path = dataset_root_folder_path + 'MashV4/ShapeNet/02828884/'

        mash_file_path_list = []

        for root, _, files in os.walk(mash_dataset_folder_path):
            for file in files:
                if not file.endswith('.npy') or file.endswith('_tmp.npy'):
                    continue

                mash_file_path = root + '/' + file

                mash_file_path_list.append(mash_file_path)

        shuffle(mash_file_path_list) 

        for mash_file_path in mash_file_path_list:
            mash = Mash.fromParamsFile(mash_file_path, device="cuda:0")

            print('mash:', mash_file_path)
            mash.renderSamplePoints()

    # view part mash folder
    if True:
        mash_folder_path = "/home/chli/chLi/Results/ma-sh/output/part_mash/"
        dataset_folder_path = '/home/chli/chLi/Dataset/MashV4/ShapeNet/'

        for root, _, files in os.walk(mash_folder_path):
            for file in files:
                if not file.endswith('.npy'):
                    continue

                mash_rel_file_path = os.path.relpath(root, mash_folder_path)

                part_mash_file_path = root + '/' + file
                mash_file_path = dataset_folder_path + mash_rel_file_path + '.npy'

                part_mash = Mash.fromParamsFile(part_mash_file_path, 90, 1000, 0.8, device="cuda")
                mash = Mash.fromParamsFile(mash_file_path, 90, 1000, 0.8, device="cuda")

                part_mash_pcd = part_mash.toSamplePcd()
                mash_pcd = mash.toSamplePcd()

                part_mash_pcd.translate([-1, 0, 0])

                print("start show mash:", mash_file_path)
                o3d.visualization.draw_geometries([part_mash_pcd, mash_pcd])

    if False:
        mash_params_folder_path = "/Volumes/chLi/Dataset/Mash/ShapeNet/mash/"

        class_name_list = os.listdir(mash_params_folder_path)

        for class_name in class_name_list:
            class_folder_path = mash_params_folder_path + class_name + "/"

            model_id_list = os.listdir(class_folder_path)

            for i, model_id in enumerate(model_id_list):
                mash_file_path = (
                    class_folder_path + model_id + "/models/model_normalized_obj.npy"
                )

                mash = Mash.fromParamsFile(mash_file_path, 10, 10000, 0.4, device="cpu")

                mash.renderSamplePoints()

                if i >= 3:
                    break

    # view single training process
    if False:
        boundary_sample_num = 36
        inner_sample_num = 100
        fps_scale = 0.8
        view_freq = 4
        show_recon_only = False

        o3d_viewer = O3DViewer()
        o3d_viewer.createWindow()

        mash_root_folder_path = "./output/"

        mash_folename_list = os.listdir(mash_root_folder_path)
        mash_folename_list.sort()

        valid_mash_folder_path_list = []

        for mash_folename in mash_folename_list:
            mash_folder_path = mash_root_folder_path + mash_folename + "/"

            if not os.path.isdir(mash_folder_path) or not os.path.exists(
                mash_folder_path
            ):
                continue

            valid_mash_folder_path_list.append(mash_folder_path)

        mash_folder_path = valid_mash_folder_path_list[-1]
        print("start view:", mash_folder_path)

        mash_filename_list = os.listdir(mash_folder_path)
        mash_filename_list.sort(key=functools.cmp_to_key(compare))

        if show_recon_only:
            mash_file_path = mash_folder_path + mash_filename_list[-1]

            mash = Mash.fromParamsFile(
                mash_file_path,
                boundary_sample_num,
                inner_sample_num,
                fps_scale,
            )
            boundary_pts, inner_pts, inner_idxs = mash.toSamplePoints()
            points = toNumpy(torch.vstack([boundary_pts, inner_pts]))

            mash.renderSamplePatches()
            return True

        for i, mash_filename in enumerate(mash_filename_list):
            if i + 1 != len(mash_filename_list):
                if (i + 1) % view_freq != 0:
                    continue

            mash_file_path = mash_folder_path + mash_filename

            mash = Mash.fromParamsFile(
                mash_file_path,
                boundary_sample_num,
                inner_sample_num,
                fps_scale,
            )
            boundary_pts, inner_pts, inner_idxs = mash.toSamplePoints()
            points = toNumpy(torch.vstack([boundary_pts, inner_pts]))

            pcd = getPointCloud(points)

            o3d_viewer.clearGeometries()
            o3d_viewer.addGeometry(pcd)

            print("now render is", i + 1, "/", len(mash_filename_list))

            o3d_viewer.update()

        o3d_viewer.run()
    return True
