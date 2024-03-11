import torch

from ma_sh.Config.constant import PI_2, PHI_WEIGHT


def toUniformSamplePhis(sample_num: int) -> torch.Tensor:
    phis_list = [PHI_WEIGHT * (i + 0.5) for i in range(sample_num)]

    phis = torch.tensor(phis_list)

    return phis


def toUniformSampleThetas(sample_num: int) -> torch.Tensor:
    cos_thetas_list = []
    for i in range(sample_num):
        cos_thetas_list.append(1.0 - (2.0 * i + 1.0) / sample_num)

    cos_thetas = torch.tensor(cos_thetas_list)

    thetas = torch.acos(cos_thetas)

    return thetas


def toMaskBoundaryPhis(anchor_num: int, mask_boundary_sample_num: int) -> torch.Tensor:
    mask_boundary_phis = torch.zeros(anchor_num, mask_boundary_sample_num)

    for i in range(mask_boundary_sample_num):
        mask_boundary_phis[:, i] = PI_2 * i / mask_boundary_sample_num

    mask_boundary_phis = mask_boundary_phis.reshape(-1)

    return mask_boundary_phis
