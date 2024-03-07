import torch

from ma_sh.Config.constant import PI_2

def toBoundIdxs(data_counts: torch.Tensor) -> torch.Tensor:
    bound_idxs = torch.zeros(data_counts.shape[0] + 1).type(data_counts.dtype).to(data_counts.device)
    for i in range(1, bound_idxs.shape[0]):
        bound_idxs[i] = data_counts[i - 1] + bound_idxs[i - 1]
    return bound_idxs

def toMaskBoundaryPhis(anchor_num: int, mask_boundary_sample_num: int) -> torch.Tensor:
    mask_boundary_phis = torch.zeros(anchor_num, mask_boundary_sample_num)

    for i in range(mask_boundary_sample_num):
        mask_boundary_phis[:, i] = PI_2 * i / mask_boundary_sample_num

    mask_boundary_phis = mask_boundary_phis.reshape(-1)

    return mask_boundary_phis
