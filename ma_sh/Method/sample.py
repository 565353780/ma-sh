import torch
from math import sqrt

from ma_sh.Config.constant import PI

def getUniformSamplePhis(point_num: int) -> torch.Tensor:
    phis_list = []
    for i in range(point_num):
        phis_list.append((2.0 * i + 1.0) / point_num - 1.0)

    phis = torch.tensor(phis_list)

    return phis

def getUniformSampleThetas(phis: torch.Tensor) -> torch.Tensor:
    weight = sqrt(phis.shape[0] * PI)

    thetas = weight * phis

    return thetas
