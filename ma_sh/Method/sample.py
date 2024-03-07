import torch
from math import sqrt, pi

def getUniformSamplePhis(point_num: int, dtype=torch.float32, device='cpu') -> torch.Tensor:
    phis = torch.tensor([(2.0 * i + 1.0) / point_num - 1.0 for i in range(point_num)]).type(dtype).to(device)

    return phis

def getUniformSampleThetas(phis: torch.Tensor) -> torch.Tensor:
    thetas = sqrt(phis.shape[0] * pi) * phis

    return thetas
