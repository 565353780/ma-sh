import torch
from math import sqrt

from ma_sh.Config.constant import PI, PI_2

def toUniformSamplePhis(point_num: int) -> torch.Tensor:
    phis_list = []
    for i in range(point_num):
        phis_list.append((2.0 * i + 1.0) / point_num - 1.0)

    phis = torch.tensor(phis_list)

    return phis

def toUniformSampleThetas(phis: torch.Tensor) -> torch.Tensor:
    weight = sqrt(phis.shape[0] * PI)

    thetas = weight * phis

    return thetas

def toMaskBoundaryPhis(anchor_num: int, mask_boundary_sample_num: int) -> torch.Tensor:
    mask_boundary_phis = torch.zeros(anchor_num, mask_boundary_sample_num)

    for i in range(mask_boundary_sample_num):
        mask_boundary_phis[:, i] = PI_2 * i / mask_boundary_sample_num

    mask_boundary_phis = mask_boundary_phis.reshape(-1)

    return mask_boundary_phis
