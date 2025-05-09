import torch

from chamfer_distance.Module.chamfer_distances import ChamferDistances

from ma_sh.Config.constant import EPSILON


def BoundaryContinuousLoss(
    anchor_num: int,
    mask_boundary_sample_points: torch.Tensor,
    mask_boundary_sample_phi_idxs: torch.Tensor,
) -> torch.Tensor:
    idx_dtype = mask_boundary_sample_phi_idxs.dtype
    dtype = mask_boundary_sample_points.dtype
    device = mask_boundary_sample_points.device

    if anchor_num < 2:
        return torch.zeros(0, dtype=dtype, device=device)

    single_boundary_sample_point_num = mask_boundary_sample_points.shape[0] // anchor_num

    other_boundary_sample_point_num = (anchor_num - 1) * single_boundary_sample_point_num

    # 将边界采样点重塑为[anchor_num, single_boundary_sample_point_num, 3]的形状
    single_mask_boundary_sample_points = mask_boundary_sample_points.view(
        anchor_num, single_boundary_sample_point_num, 3)

    # 创建点数据索引矩阵
    point_data_idx_matrix = torch.arange(0, mask_boundary_sample_points.size(0), dtype=idx_dtype, device=device)
    point_data_idx_matrix = point_data_idx_matrix.view(anchor_num, single_boundary_sample_point_num)

    # 创建数据行索引
    data_row_idx = torch.arange(1, anchor_num + 1, dtype=idx_dtype, device=device)

    # 创建排除矩阵，用于排除自身锚点
    exclusion_matrix = data_row_idx.view(1, anchor_num).repeat(anchor_num, 1)
    exclusion_matrix.fill_diagonal_(0)
    exclusion_matrix = exclusion_matrix.view(-1)

    # 获取非零行索引
    data_row_idxs = exclusion_matrix.index_select(0, torch.nonzero(exclusion_matrix).view(-1)) - 1

    # 获取点数据索引
    point_data_idxs = point_data_idx_matrix.index_select(0, data_row_idxs).view(-1)

    # 获取其他锚点的边界采样点
    other_mask_boundary_sample_points = mask_boundary_sample_points.index_select(
        0, point_data_idxs).view(anchor_num, other_boundary_sample_point_num, 3)

    # 计算Chamfer距离
    boundary_chamfer_distances = ChamferDistances.namedAlgo('cuda')(
        single_mask_boundary_sample_points, other_mask_boundary_sample_points)

    # 获取边界连接距离的平方
    boundary_connect_dists2 = boundary_chamfer_distances[0]

    # 计算边界连接距离
    boundary_connect_dists = torch.sqrt(boundary_connect_dists2 + EPSILON)

    # 计算边界连接损失
    boundary_connect_loss = torch.mean(boundary_connect_dists)

    return boundary_connect_loss
