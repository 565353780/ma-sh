import torch

from chamfer_distance.Module.chamfer_distances import ChamferDistances

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

    # 构造 [A, (A-1)*P, 3] 的 other_points，保证每个 anchor 对应其他 anchor 的点集
    # 避免使用 expand 操作创建大张量，这会在 anchor_num 较大时导致显存溢出
    other_points = torch.empty(
        anchor_num, (anchor_num - 1) * P, 3, dtype=dtype, device=device
    )

    for i in range(anchor_num):
        # 选出除第 i 个 anchor 外的所有点（zero-copy 子 view）
        if i == 0:
            others = boundary_points[1:].reshape(-1, 3)
        elif i == anchor_num - 1:
            others = boundary_points[:i].reshape(-1, 3)
        else:
            others = torch.cat(
                [boundary_points[:i], boundary_points[i + 1 :]], dim=0
            ).reshape(-1, 3)
        other_points[i] = others.view((anchor_num - 1) * P, 3)  # 赋值到 batch 中

    dist1 = ChamferDistances.namedAlgo("cuda")(
        boundary_points,
        other_points,  # [A, P, 3] vs [A, (A-1)*P, 3]
    )[0]

    dist = torch.sqrt(dist1 + EPSILON)
    loss = dist.mean()
    return loss
