import gc
import torch

from chamfer_distance.Module.chamfer_distances import ChamferDistances

from ma_sh.Config.constant import EPSILON


def BoundaryContinuousLoss(
    anchor_num: int,
    mask_boundary_sample_points: torch.Tensor,
    mask_boundary_sample_phi_idxs: torch.Tensor,
) -> torch.Tensor:
    """
    计算边界连续性损失，同时优化GPU内存使用。
    
    该函数实现了边界连续性损失的计算，并采用了多种内存优化措施：
    1. 使用torch.where代替torch.nonzero减少内存使用
    2. 及时释放中间张量减少内存占用
    3. 使用detach_()减少不必要的梯度计算内存
    4. 在函数结束时清理CUDA缓存
    5. 优化异常处理确保资源释放
    
    Args:
        anchor_num: 锚点数量
        mask_boundary_sample_points: 掩码边界采样点
        mask_boundary_sample_phi_idxs: 掩码边界采样phi索引
        
    Returns:
        边界连接损失张量
    """
    # 初始化结果变量为None，用于在finally块中检查是否需要返回结果
    result = None
    
    # 获取输入张量的数据类型和设备信息
    idx_dtype = mask_boundary_sample_phi_idxs.dtype
    dtype = mask_boundary_sample_points.dtype
    device = mask_boundary_sample_points.device

    # 如果锚点数量小于2，直接返回零张量
    if anchor_num < 2:
        return torch.zeros(0, dtype=dtype, device=device)

    # 计算每个锚点的边界采样点数量
    single_boundary_sample_point_num = mask_boundary_sample_points.shape[0] // anchor_num
    # 计算其他锚点的边界采样点总数
    other_boundary_sample_point_num = (anchor_num - 1) * single_boundary_sample_point_num

    try:
        # 将边界采样点重塑为[anchor_num, single_boundary_sample_point_num, 3]的形状
        # 使用contiguous()确保内存布局连续，减少内存碎片
        single_mask_boundary_sample_points = mask_boundary_sample_points.view(
            anchor_num, single_boundary_sample_point_num, 3).contiguous()

        # 创建点数据索引矩阵
        point_data_idx_matrix = torch.arange(0, mask_boundary_sample_points.size(0), dtype=idx_dtype, device=device)
        point_data_idx_matrix = point_data_idx_matrix.view(anchor_num, single_boundary_sample_point_num)

        # 创建数据行索引
        data_row_idx = torch.arange(1, anchor_num + 1, dtype=idx_dtype, device=device)

        # 创建排除矩阵，用于排除自身锚点
        exclusion_matrix = data_row_idx.view(1, anchor_num).repeat(anchor_num, 1)
        exclusion_matrix.fill_diagonal_(0)
        exclusion_matrix = exclusion_matrix.view(-1)

        # 获取非零行索引 - 使用torch.where代替torch.nonzero以减少内存使用
        non_zero_indices = torch.where(exclusion_matrix != 0)[0]
        data_row_idxs = exclusion_matrix.index_select(0, non_zero_indices) - 1
        
        # 获取点数据索引
        point_data_idxs = point_data_idx_matrix.index_select(0, data_row_idxs).view(-1)

        # 获取其他锚点的边界采样点
        # 使用contiguous()确保内存布局连续，减少内存碎片
        other_mask_boundary_sample_points = mask_boundary_sample_points.index_select(
            0, point_data_idxs).view(anchor_num, other_boundary_sample_point_num, 3).contiguous()
        
        # 确保索引张量不再需要时被释放
        del non_zero_indices, data_row_idxs, point_data_idxs
        
        try:
            # 计算Chamfer距离 - 保留梯度计算但优化内存使用
            boundary_chamfer_distances = ChamferDistances.namedAlgo('cuda')(
                single_mask_boundary_sample_points, other_mask_boundary_sample_points)
            
            # 获取边界连接距离的平方
            boundary_connect_dists2 = boundary_chamfer_distances[0]
            
            # 立即释放不再需要的张量
            # 只保留需要的结果，释放其他部分
            if len(boundary_chamfer_distances) > 1:
                for i in range(1, len(boundary_chamfer_distances)):
                    if isinstance(boundary_chamfer_distances[i], torch.Tensor):
                        # 使用detach_()和_的操作可以减少内存占用
                        boundary_chamfer_distances[i].detach_()
                        
            # 释放不再需要的输入张量
            del single_mask_boundary_sample_points, other_mask_boundary_sample_points
            
            # 计算边界连接距离
            boundary_connect_dists = torch.sqrt(boundary_connect_dists2 + EPSILON)
            
            # 释放不再需要的张量
            del boundary_connect_dists2, boundary_chamfer_distances
            
            # 计算边界连接损失
            boundary_connect_loss = torch.mean(boundary_connect_dists)
            
            # 复制结果并释放中间变量
            result = boundary_connect_loss.clone()
            del boundary_connect_dists, boundary_connect_loss
            
        except Exception as e:
            # 捕获内部计算过程中的异常
            print(f"[ERROR][BoundaryContinuousLoss] 计算过程出错: {str(e)}")
            # 重新抛出异常，让外层try-finally处理清理工作
            raise e
    
    finally:
        # 显式清除中间变量以防止GPU内存泄漏
        # 在函数结束时清理临时变量
        local_vars = list(locals().items())
        for var_name, var_val in local_vars:
            if var_name not in ['result'] and isinstance(var_val, torch.Tensor):
                del var_val
        
        # 强制Python垃圾回收
        gc.collect()
        
        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 如果计算成功完成，返回结果；否则返回零张量（表示计算失败）
    if result is not None:
        return result
    else:
        print("[WARN][BoundaryContinuousLoss] 计算失败，返回零张量")
        return torch.zeros(0, dtype=dtype, device=device)
