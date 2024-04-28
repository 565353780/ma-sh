import torch
import mash_cpp

from ma_sh.Config.constant import EPSILON

def toAnchorCoverageLoss(anchor_num, boundary_pts, inner_pts, mask_boundary_phi_idxs, inner_idxs, gt_points) -> torch.Tensor:
    sample_points = torch.vstack([boundary_pts, inner_pts]).unsqueeze(0).type(gt_points.dtype)

    fd2, cd2, fi, ci = mash_cpp.toChamferDistance(sample_points, gt_points)

    fd2 = fd2.squeeze(0)
    cd2 = cd2.squeeze(0)
    fi = fi.squeeze(0)
    ci = ci.squeeze(0)

    mask_boundary_sample_num = boundary_pts.shape[0]

    cd = torch.sqrt(cd2 + EPSILON)

    pull_boundary_mask = ci < mask_boundary_sample_num

    pull_boundary_idxs = torch.where(pull_boundary_mask)[0]

    pull_inner_idxs = torch.where(~pull_boundary_mask)[0]

    pull_boundary_point_idxs = ci[pull_boundary_idxs]

    pull_inner_point_idxs = ci[pull_inner_idxs] - mask_boundary_sample_num

    pull_boundary_anchor_idxs = mask_boundary_phi_idxs[pull_boundary_point_idxs]

    pull_inner_anchor_idxs = inner_idxs[pull_inner_point_idxs]

    coverage_loss = torch.tensor(0.0).type(gt_points.dtype).to(gt_points.device)

    assert not torch.isnan(coverage_loss).any(), 'initial coverage_loss failed!'

    for i in range(anchor_num):
        current_pull_boundary_idxs = torch.where(pull_boundary_anchor_idxs == i)[0]
        current_pull_inner_idxs = torch.where(pull_inner_anchor_idxs == i)[0] + mask_boundary_sample_num

        current_pull_idxs = torch.hstack([current_pull_boundary_idxs, current_pull_inner_idxs])

        current_coverage_dists = cd[current_pull_idxs]

        assert not torch.isnan(current_coverage_dists).any()

        current_mean_cd = torch.mean(current_coverage_dists)

        assert not torch.isnan(current_mean_cd).any(), 'anchor' + str(i) + ' mean cd makes coverage_loss be nan!' + str(current_pull_idxs.shape)

        coverage_loss = coverage_loss + current_mean_cd

        assert not torch.isnan(coverage_loss).any(), 'anchor' + str(i) + ' makes coverage_loss be nan!' + str(current_pull_idxs.shape)

    coverage_loss = coverage_loss / float(anchor_num)

    assert not torch.isnan(coverage_loss).any(), 'divide anchor_num makes nan!'

    return coverage_loss
