import sys
sys.path.append('../param-gauss-recon')

import os
import torch
import numpy as np
import open3d as o3d
from shutil import copyfile
from tqdm import tqdm, trange

import mash_cpp

from param_gauss_recon.Module.reconstructor import Reconstructor

from ma_sh.Method.data import toNumpy
from ma_sh.Method.pcd import getPointCloud
from ma_sh.Method.render import renderGeometries

@torch.no_grad()
def toNormalTagsWithCluster(anchor_num: int, points: torch.Tensor,
                point_idxs: torch.Tensor,
                normals: torch.Tensor,
                fps_sample_scale: float = -1) -> torch.Tensor:
    normal_tags = torch.ones([anchor_num], dtype=points.dtype).to(points.device)

    tagged_anchor_idxs = [0]
    untagged_anchor_idxs = list(range(1, anchor_num))

    points_list = []
    normals_list = []

    print('[INFO][normal::toNormalTagsWithCluster]')
    print('\t start collect anchor points and normals...')
    for i in trange(anchor_num):
        anchor_point_mask = point_idxs == i

        anchor_points = points[anchor_point_mask]
        anchor_normals = normals[anchor_point_mask]

        normal_norms = torch.norm(anchor_normals, dim=1)

        valid_normal_mask = normal_norms > 0

        valid_anchor_points = anchor_points[valid_normal_mask]
        valid_anchor_normals = anchor_normals[valid_normal_mask]

        if fps_sample_scale > 0:
            fps_idxs = mash_cpp.toFPSPointIdxs(valid_anchor_points, torch.zeros([valid_anchor_points.shape[0]]).type(torch.int), fps_sample_scale, 1)

            fps_points = valid_anchor_points[fps_idxs]
            fps_normals = valid_anchor_normals[fps_idxs]

            points_list.append(fps_points)
            normals_list.append(fps_normals)
        else:
            points_list.append(valid_anchor_points)
            normals_list.append(valid_anchor_normals)

    print('[INFO][normal::toNormalTagsWithCluster]')
    print('\t start auto estimate normal tags for each anchor...')
    pbar = tqdm(total=anchor_num)
    pbar.update(1)
    while len(tagged_anchor_idxs) < anchor_num:
        tagged_points = [points_list[i] for i in tagged_anchor_idxs]
        untagged_points = [points_list[i] for i in untagged_anchor_idxs]

        tagged_normals = [normals_list[i] for i in tagged_anchor_idxs]
        untagged_normals = [normals_list[i] for i in untagged_anchor_idxs]

        tagged_idxs = mash_cpp.toIdxs(mash_cpp.toCounts(tagged_points)).type(torch.int)
        untagged_idxs = mash_cpp.toIdxs(mash_cpp.toCounts(untagged_points)).type(torch.int)

        if len(tagged_points) == 1:
            tagged_pts = tagged_points[0]
        else:
            tagged_pts = torch.vstack(tagged_points)
        if len(untagged_points) == 1:
            untagged_pts = untagged_points[0]
        else:
            untagged_pts = torch.vstack(untagged_points)

        if len(tagged_normals) == 1:
            tagged_ns = tagged_normals[0]
        else:
            tagged_ns = torch.vstack(tagged_normals)
        if len(untagged_normals) == 1:
            untagged_ns = untagged_normals[0]
        else:
            untagged_ns = torch.vstack(untagged_normals)

        if False:
            pcd1 = getPointCloud(toNumpy(tagged_pts), toNumpy(tagged_ns))
            pcd2 = getPointCloud(toNumpy(untagged_pts), toNumpy(untagged_ns))
            pcd2.translate([0, 0, 1])

            o3d.visualization.draw_geometries([pcd1, pcd2], point_show_normal=True)

        dists1, dists2, idxs1, idxs2 = mash_cpp.toChamferDistance(tagged_pts.unsqueeze(0), untagged_pts.unsqueeze(0))

        min_dist1_idx = torch.argmin(dists1)
        min_dist2_idx = torch.argmin(dists2)

        min_dist_tagged_pts_idx = idxs2[0, min_dist2_idx]
        min_dist_untagged_pts_idx = idxs1[0, min_dist1_idx]

        min_dist_tagged_idx = tagged_anchor_idxs[tagged_idxs[min_dist_tagged_pts_idx]]
        min_dist_untagged_idx = untagged_anchor_idxs[untagged_idxs[min_dist_untagged_pts_idx]]

        min_dist_tagged_point = tagged_pts[min_dist_tagged_pts_idx]
        min_dist_untagged_point = untagged_pts[min_dist_untagged_pts_idx]

        min_dist_tagged_normal = tagged_ns[min_dist_tagged_pts_idx]
        min_dist_untagged_normal = untagged_ns[min_dist_untagged_pts_idx]

        normal_angle = torch.acos(min_dist_tagged_normal.dot(min_dist_untagged_normal))

        if normal_angle < np.pi / 2.0:
            normal_tags[min_dist_untagged_idx] = normal_tags[min_dist_tagged_idx]
        else:
            normal_tags[min_dist_untagged_idx] = -1.0 * normal_tags[min_dist_tagged_idx]

        tagged_anchor_idxs.append(min_dist_untagged_idx)
        untagged_anchor_idxs.remove(min_dist_untagged_idx)

        if False:
            total_pcd = getPointCloud(points.cpu().numpy())
            total_pcd.translate([0, 0, 1])

            pcd = getPointCloud(torch.vstack([min_dist_tagged_point, min_dist_untagged_point]).cpu().numpy())
            pcd.normals = o3d.utility.Vector3dVector(torch.vstack([min_dist_tagged_normal, min_dist_untagged_normal]).cpu().numpy())
            o3d.visualization.draw_geometries([total_pcd, pcd], point_show_normal=True)

        pbar.update(1)

    return normal_tags

@torch.no_grad()
def toNormalTagsWithIterativeCluster(anchor_num: int, points: torch.Tensor,
                point_idxs: torch.Tensor,
                normals: torch.Tensor,
                fps_sample_scale: float = -1) -> torch.Tensor:
    iter_normals = normals.detach().clone()

    normal_tagss = toNormalTagsWithCluster(anchor_num, points, point_idxs, iter_normals, fps_sample_scale)

    finished = (normal_tagss == 1.0).all()

    while not finished:
        for i in range(anchor_num):
            anchor_normal_idxs = point_idxs == i
            iter_normals[anchor_normal_idxs] *= normal_tagss[i]

        normal_tagss = toNormalTagsWithCluster(anchor_num, points, point_idxs, iter_normals, fps_sample_scale)

        finished = (normal_tagss == 1.0).all()
    return normal_tagss

@torch.no_grad()
def toNormalTagWithPGR(anchor_num: int, points: torch.Tensor,
                point_idxs: torch.Tensor,
                normals: torch.Tensor,
                fps_sample_scale: float = -1) -> torch.Tensor:
    tmp_save_folder_path = './tmp/normal/'
    os.makedirs(tmp_save_folder_path, exist_ok=True)

    normal_tags = torch.ones([anchor_num], dtype=points.dtype).to(points.device)

    points_list = []
    normals_list = []

    print('[INFO][normal::toNormalTagWithPGR]')
    print('\t start collect anchor points and normals...')
    for i in trange(anchor_num):
        anchor_point_mask = point_idxs == i

        anchor_points = points[anchor_point_mask]
        anchor_normals = normals[anchor_point_mask]

        normal_norms = torch.norm(anchor_normals, dim=1)

        valid_normal_mask = normal_norms > 0

        valid_anchor_points = anchor_points[valid_normal_mask]
        valid_anchor_normals = anchor_normals[valid_normal_mask]

        if fps_sample_scale > 0:
            fps_idxs = mash_cpp.toFPSPointIdxs(valid_anchor_points, torch.zeros([valid_anchor_points.shape[0]]).type(torch.int).to(points.device), fps_sample_scale, 1)

            fps_points = valid_anchor_points[fps_idxs]
            fps_normals = valid_anchor_normals[fps_idxs]

            points_list.append(fps_points)
            normals_list.append(fps_normals)
        else:
            points_list.append(valid_anchor_points)
            normals_list.append(valid_anchor_normals)

    mash_points = torch.vstack(points_list)

    tmp_save_pcd_file_path = tmp_save_folder_path + 'pcd.xyz'
    pcd = getPointCloud(toNumpy(mash_points))
    o3d.io.write_point_cloud(tmp_save_pcd_file_path, pcd, write_ascii=True)

    reconstructor = Reconstructor()
    reconstructor.reconstructSurface(
        input=tmp_save_pcd_file_path,
        sample_point_num=None,
        width_k=7,
        width_max=0.015,
        width_min=0.0015,
        alpha=1.08,
        max_iters=None,
        max_depth=10,
        min_depth=1,
        cpu=not torch.cuda.is_available(),
        save_r=None,
        recon_mesh=False,
    )

    param_midfix = reconstructor.param_midfix

    pgr_pcd_file_path = '../ma-sh/results/pcd/solve/pcd' + param_midfix + 'lse.xyz'

    if not os.path.exists(pgr_pcd_file_path):
        print('[ERROR][normal::toNormalTagWithPGR]')
        print('\t reconstructSurface failed! will return unrefined normal tags!')
        return normal_tags

    pgr_pcd_with_normal_file_path = '../ma-sh/results/pcd/solve/pcd' + param_midfix + 'lse.xyzn'
    copyfile(pgr_pcd_file_path, pgr_pcd_with_normal_file_path)
    pgr_pcd = o3d.io.read_point_cloud(pgr_pcd_with_normal_file_path)

    pgr_points = torch.from_numpy(np.asarray(pgr_pcd.points)).type(points.dtype).to(points.device)

    mash_bound_min = torch.min(mash_points, dim=0)[0]
    mash_bound_max = torch.max(mash_points, dim=0)[0]
    pgr_bound_min = torch.min(pgr_points, dim=0)[0]
    pgr_bound_max = torch.max(pgr_points, dim=0)[0]

    mash_translate = (mash_bound_min + mash_bound_max) / 2.0
    mash_scale = mash_bound_max - mash_bound_min
    pgr_translate = (pgr_bound_min + pgr_bound_max) / 2.0
    pgr_scale = pgr_bound_max - pgr_bound_min

    pgr_points = (pgr_points - pgr_translate) / pgr_scale * mash_scale + mash_translate

    pgr_normals = torch.from_numpy(np.asarray(pgr_pcd.normals)).to(points.device)
    pgr_normal_norms = torch.norm(pgr_normals, dim=1)
    valid_pgr_normal_mask = pgr_normal_norms > 0

    valid_pgr_points = pgr_points[valid_pgr_normal_mask]
    valid_pgr_normals = pgr_normals[valid_pgr_normal_mask] / pgr_normal_norms[valid_pgr_normal_mask].reshape(-1, 1)

    _, _, idxs1, _ = mash_cpp.toChamferDistance(mash_points.unsqueeze(0), valid_pgr_points.unsqueeze(0))

    mash_normals = torch.vstack(normals_list)

    if False:
        mash_pcd = getPointCloud(toNumpy(mash_points), toNumpy(mash_normals))
        pgr_pcd = getPointCloud(toNumpy(pgr_points), toNumpy(pgr_normals))
        pgr_pcd.translate([0, 1.0, 0])

        renderGeometries([mash_pcd, pgr_pcd], "Mash and PGR normals", True)

    anchor_idxs = mash_cpp.toIdxs(mash_cpp.toCounts(points_list)).type(torch.int)
    for i in range(anchor_num):
        anchor_mask = anchor_idxs == i

        anchor_points = mash_points[anchor_mask]
        anchor_normals = mash_normals[anchor_mask]

        pgr_idxs = idxs1[0, anchor_mask]

        anchor_pgr_points = valid_pgr_points[pgr_idxs]
        anchor_pgr_normals = valid_pgr_normals[pgr_idxs]

        if False:
            anchor_mash_pcd = getPointCloud(toNumpy(anchor_points), toNumpy(anchor_normals))
            anchor_pgr_pcd = getPointCloud(toNumpy(anchor_pgr_points), toNumpy(anchor_pgr_normals))
            anchor_pgr_pcd.translate([0, 0.2, 0])

            renderGeometries([anchor_mash_pcd, anchor_pgr_pcd], "Mash and PGR Anchor normals", True)

        same_direction_dists = torch.norm(anchor_normals - anchor_pgr_normals, dim=1)
        opposite_direction_dists = torch.norm(anchor_normals + anchor_pgr_normals, dim=1)

        same_direction_dist = torch.mean(same_direction_dists)
        opposite_direction_dist = torch.mean(opposite_direction_dists)

        if same_direction_dist > opposite_direction_dist:
            normal_tags[i] = -1.0

    return normal_tags

def toNormalTags(anchor_num: int, points: torch.Tensor,
                point_idxs: torch.Tensor,
                normals: torch.Tensor,
                fps_sample_scale: float = -1,
                 mode: str = 'pgr') -> torch.Tensor:
    valid_modes = ['cluster', 'iter_cluster', 'pgr']

    if mode == 'cluster':
        return toNormalTagsWithCluster(anchor_num, points, point_idxs, normals, fps_sample_scale)
    elif mode == 'iter_cluster':
        return toNormalTagsWithIterativeCluster(anchor_num, points, point_idxs, normals, fps_sample_scale)
    elif mode == 'pgr':
        return toNormalTagWithPGR(anchor_num, points, point_idxs, normals, fps_sample_scale)

    print('[ERROR][normal::toNormalTags]')
    print('\t mode not valid! will return unrefined normal tags!')
    print('\t valid modes:', valid_modes)
    return torch.ones([anchor_num], dtype=points.dtype).to(points.device)
