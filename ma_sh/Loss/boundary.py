import torch

from chamfer_distance.Module.sided_distances import SidedDistances

from ma_sh.Config.constant import EPSILON


def BoundaryContinuousLoss(
    anchor_num: int,
    mask_boundary_sample_points: torch.Tensor,
) -> torch.Tensor:
    if anchor_num < 2:
        return torch.zeros(1, device=mask_boundary_sample_points.device)

    device = mask_boundary_sample_points.device
    dtype = mask_boundary_sample_points.dtype

    # 每个 anchor 的采样点数量
    P = mask_boundary_sample_points.shape[0] // anchor_num

    # [A, P, 3]：每个 anchor 的点，zero-copy reshape
    boundary_points = mask_boundary_sample_points.view(anchor_num, P, 3)

    # 构建 [A, A-1, P, 3] 的“其他 anchor 视图”
    # 先构建 A x A x P x 3，再mask掉对角线
    expanded = boundary_points.unsqueeze(0).expand(
        anchor_num, anchor_num, P, 3
    )  # [A, A, P, 3]
    other_points = expanded[
        ~torch.eye(anchor_num, dtype=torch.bool, device=device)
    ]  # [(A * (A - 1)), P, 3]
    other_points = other_points.view(
        anchor_num, (anchor_num - 1) * P, 3
    )  # [A, (A-1)*P, 3]

    # 计算 Chamfer 距离（建议使用 batch 版本的 chamfer kernel）
    try:
        chamfer_dist = SidedDistances.namedAlgo("cuda")(
            boundary_points,
            other_points,  # [A, P, 3] vs [A, (A-1)*P, 3]
        )
        dist2 = chamfer_dist[0]  # [A, P]

        dist = torch.sqrt(dist2 + EPSILON)
        loss = dist.mean()
        return loss

    except Exception as e:
        print(f"[ERROR] Chamfer distance error: {str(e)}")
        return torch.zeros(1, dtype=dtype, device=device)
