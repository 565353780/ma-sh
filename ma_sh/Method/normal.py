import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm, trange
from math import sqrt

import mash_cpp

from ma_sh.Method.data import toNumpy
from ma_sh.Method.pcd import getPointCloud

@torch.no_grad()
def toNormalTag(anchor_num: int, mask_boundary_points: torch.Tensor,
                mask_boundary_point_idxs: torch.Tensor,
                mask_boundary_normals: torch.Tensor,
                fps_sample_num: int = -1) -> torch.Tensor:
    normal_tag = torch.ones([anchor_num], dtype=mask_boundary_points.dtype).to(mask_boundary_points.device)

    tagged_anchor_idxs = [0]
    untagged_anchor_idxs = list(range(1, anchor_num))

    mask_boundary_points_list = []
    mask_boundary_normals_list = []

    for i in trange(anchor_num):
        anchor_point_mask = mask_boundary_point_idxs == i

        anchor_boundary_points = mask_boundary_points[anchor_point_mask]
        anchor_boundary_normals = mask_boundary_normals[anchor_point_mask]

        normal_norms = torch.norm(anchor_boundary_normals, dim=1)

        valid_normal_mask = normal_norms > 0

        valid_anchor_boundary_points = anchor_boundary_points[valid_normal_mask]
        valid_anchor_boundary_normals = anchor_boundary_normals[valid_normal_mask]

        if fps_sample_num > 0:
            fps_scale = fps_sample_num / valid_anchor_boundary_points.shape[0]
            fps_idxs = mash_cpp.toFPSPointIdxs(valid_anchor_boundary_points, torch.zeros([valid_anchor_boundary_points.shape[0]]).type(torch.int), fps_scale, 1)

            fps_points = valid_anchor_boundary_points[fps_idxs]
            fps_normals = valid_anchor_boundary_normals[fps_idxs]

            mask_boundary_points_list.append(fps_points)
            mask_boundary_normals_list.append(fps_normals)
        else:
            mask_boundary_points_list.append(valid_anchor_boundary_points)
            mask_boundary_normals_list.append(valid_anchor_boundary_normals)

    print('[INFO][normal::toNormalTag]')
    print('\t start auto estimate normal tags for each anchor...')
    pbar = tqdm(total=anchor_num)
    pbar.update(1)
    while len(tagged_anchor_idxs) < anchor_num:
        tagged_points = [mask_boundary_points_list[i] for i in tagged_anchor_idxs]
        untagged_points = [mask_boundary_points_list[i] for i in untagged_anchor_idxs]

        tagged_normals = [mask_boundary_normals_list[i] for i in tagged_anchor_idxs]
        untagged_normals = [mask_boundary_normals_list[i] for i in untagged_anchor_idxs]

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

        point_dist = torch.norm(min_dist_tagged_point - min_dist_untagged_point) ** 2
        normal_angle = torch.acos(min_dist_tagged_normal.dot(min_dist_untagged_normal))

        if normal_angle < np.pi / 2.0:
            normal_tag[min_dist_untagged_idx] = normal_tag[min_dist_tagged_idx]
        else:
            normal_tag[min_dist_untagged_idx] = -1.0 * normal_tag[min_dist_tagged_idx]

        tagged_anchor_idxs.append(min_dist_untagged_idx)
        untagged_anchor_idxs.remove(min_dist_untagged_idx)

        if False:
            total_pcd = getPointCloud(mask_boundary_points.cpu().numpy())
            total_pcd.translate([0, 0, 1])

            pcd = getPointCloud(torch.vstack([min_dist_tagged_point, min_dist_untagged_point]).cpu().numpy())
            pcd.normals = o3d.utility.Vector3dVector(torch.vstack([min_dist_tagged_normal, min_dist_untagged_normal]).cpu().numpy())
            o3d.visualization.draw_geometries([total_pcd, pcd], point_show_normal=True)

        pbar.update(1)

    return normal_tag
