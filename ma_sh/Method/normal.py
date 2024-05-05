import torch
import open3d as o3d
from tqdm import tqdm
from math import sqrt

import mash_cpp

from ma_sh.Method.pcd import getPointCloud

@torch.no_grad()
def toNormalTag(anchor_num: int, mask_boundary_points: torch.Tensor,
                mask_boundary_point_idxs: torch.Tensor,
                mask_boundary_normals: torch.Tensor) -> torch.Tensor:
    normal_tag = torch.ones([anchor_num], dtype=torch.bool).to(mask_boundary_points.device)

    tagged_anchor_idxs = [0]
    untagged_anchor_idxs = list(range(1, anchor_num))

    mask_boundary_points_list = []
    mask_boundary_normals_list = []

    for i in range(anchor_num):
        anchor_point_mask = mask_boundary_point_idxs == i

        anchor_boundary_points = mask_boundary_points[anchor_point_mask]
        anchor_boundary_normals = mask_boundary_normals[anchor_point_mask]

        normal_norms = torch.norm(anchor_boundary_normals, dim=1)

        valid_normal_mask = normal_norms > 0

        valid_anchor_boundary_points = anchor_boundary_points[valid_normal_mask]
        valid_anchor_boundary_normals = anchor_boundary_normals[valid_normal_mask]

        mask_boundary_points_list.append(valid_anchor_boundary_points)
        mask_boundary_normals_list.append(valid_anchor_boundary_normals)

    pbar = tqdm(total=anchor_num)
    pbar.update(1)
    while len(tagged_anchor_idxs) < anchor_num:
        min_dist_tagged_point_idx = None
        min_dist_untagged_point_idx = None
        min_dist = float('inf')

        for i, tagged_anchor_idx in enumerate(tagged_anchor_idxs):
            tagged_boundary_points = mask_boundary_points_list[tagged_anchor_idx]

            for j, untagged_anchor_idx in enumerate(untagged_anchor_idxs):
                untagged_boundary_points = mask_boundary_points_list[untagged_anchor_idx]

                dists1, dists2, idxs1, idxs2 = mash_cpp.toChamferDistance(tagged_boundary_points.unsqueeze(0), untagged_boundary_points.unsqueeze(0))

                min_dist1_idx = torch.argmin(dists1)
                min_dist2_idx = torch.argmin(dists2)

                current_min_dist_tagged_point_idx = idxs2[0, min_dist2_idx]
                current_min_dist_untagged_point_idx = idxs1[0, min_dist1_idx]

                current_min_dist = dists1[0, min_dist1_idx]

                if current_min_dist < min_dist:
                    min_dist = current_min_dist
                    min_dist_tagged_point_idx = [tagged_anchor_idx, current_min_dist_tagged_point_idx]
                    min_dist_untagged_point_idx = [untagged_anchor_idx, current_min_dist_untagged_point_idx]

        tagged_point = mask_boundary_points_list[min_dist_tagged_point_idx[0]][min_dist_tagged_point_idx[1]]
        untagged_point = mask_boundary_points_list[min_dist_untagged_point_idx[0]][min_dist_untagged_point_idx[1]]

        tagged_normal = mask_boundary_normals_list[min_dist_tagged_point_idx[0]][min_dist_tagged_point_idx[1]]
        untagged_normal = mask_boundary_normals_list[min_dist_untagged_point_idx[0]][min_dist_untagged_point_idx[1]]

        point_dist = torch.norm(tagged_point - untagged_point) ** 2
        normal_dist = torch.norm(tagged_normal - untagged_normal)

        min_dist_tagged_points_idx = min_dist_tagged_point_idx[0]
        min_dist_untagged_points_idx = min_dist_untagged_point_idx[0]
        if normal_dist < sqrt(2.0):
            normal_tag[min_dist_untagged_points_idx] = normal_tag[min_dist_tagged_points_idx]
        else:
            normal_tag[min_dist_untagged_points_idx] = not normal_tag[min_dist_tagged_points_idx]
        tagged_anchor_idxs.append(min_dist_untagged_points_idx)
        untagged_anchor_idxs.remove(min_dist_untagged_points_idx)

        if False:
            total_pcd = getPointCloud(mask_boundary_points.cpu().numpy())
            total_pcd.translate([0, 0, 1])

            pcd = getPointCloud(torch.vstack([tagged_point, untagged_point]).cpu().numpy())
            pcd.normals = o3d.utility.Vector3dVector(torch.vstack([tagged_normal, untagged_normal]).cpu().numpy())
            o3d.visualization.draw_geometries([total_pcd, pcd], point_show_normal=True)
            exit()

        pbar.update(1)

    return normal_tag
