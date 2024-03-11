import numpy
import torch
import math

from ma_sh.Config.weights import W0, W1, W2, W3, W4, W5, W6


def toSHWeight(degree, real_idx):
    match degree:
        case 0:
            return W0[real_idx]
        case 1:
            return W1[real_idx]
        case 2:
            return W2[real_idx]
        case 3:
            return W3[real_idx]
        case 4:
            return W4[real_idx]
        case 5:
            return W5[real_idx]
        case 6:
            return W6[real_idx]


def toSHCommonValue(idx, phi, theta):
    if idx == 0:
        return 1.0

    st = torch.sin(theta) ** abs(idx)

    if idx > 0:
        return torch.cos(1.0 * idx * phi) * st

    return torch.sin(-1.0 * idx * phi) * st


def toDeg1ThetaValue(real_idx, theta):
    match real_idx:
        case 0:
            return torch.cos(theta)
        case 1:
            return 1.0


def toDeg2ThetaValue(real_idx, theta):
    match real_idx:
        case 0:
            ct = torch.cos(theta)
            return 3.0 * ct * ct - 1.0
        case 1:
            return torch.cos(theta)
        case 2:
            return 1.0


def toDeg3ThetaValue(real_idx, theta):
    match real_idx:
        case 0:
            ct = torch.cos(theta)
            return (5.0 * ct * ct - 3.0) * ct
        case 1:
            ct = torch.cos(theta)
            return 5.0 * ct * ct - 1.0
        case 2:
            return torch.cos(theta)
        case 3:
            return 1.0


def toDeg4ThetaValue(real_idx, theta):
    match real_idx:
        case 0:
            ct = torch.cos(theta)
            return (35.0 * ct * ct - 30.0) * ct * ct + 3.0
        case 1:
            ct = torch.cos(theta)
            return (7.0 * ct * ct - 3.0) * ct
        case 2:
            ct = torch.cos(theta)
            return 7.0 * ct * ct - 1.0
        case 3:
            return torch.cos(theta)
        case 4:
            return 1.0


def toDeg5ThetaValue(real_idx, theta):
    match real_idx:
        case 0:
            ct = torch.cos(theta)
            return ((63.0 * ct * ct - 70.0) * ct * ct + 15.0) * ct
        case 1:
            ct = torch.cos(theta)
            return (21.0 * ct * ct - 14.0) * ct * ct + 1.0
        case 2:
            ct = torch.cos(theta)
            return (3.0 * ct * ct - 1.0) * ct
        case 3:
            ct = torch.cos(theta)
            return 9.0 * ct * ct - 1.0
        case 4:
            return torch.cos(theta)
        case 5:
            return 1.0


def toDeg6ThetaValue(real_idx, theta):
    match real_idx:
        case 0:
            ct = torch.cos(theta)
            return ((231.0 * ct * ct - 315.0) * ct * ct + 105.0) * ct * ct - 5.0
        case 1:
            ct = torch.cos(theta)
            return ((33.0 * ct * ct - 30.0) * ct * ct + 5.0) * ct
        case 2:
            ct = torch.cos(theta)
            return (33.0 * ct * ct - 18.0) * ct * ct + 1.0
        case 3:
            ct = torch.cos(theta)
            return (11.0 * ct * ct - 3.0) * ct
        case 4:
            ct = torch.cos(theta)
            return 11.0 * ct * ct - 1.0
        case 5:
            return torch.cos(theta)
        case 6:
            return 1.0


def toSHResValue(degree, real_idx, theta):
    match degree:
        case 0:
            return torch.ones_like(theta)
        case 1:
            return toDeg1ThetaValue(real_idx, theta)
        case 2:
            return toDeg2ThetaValue(real_idx, theta)
        case 3:
            return toDeg3ThetaValue(real_idx, theta)
        case 4:
            return toDeg4ThetaValue(real_idx, theta)
        case 5:
            return toDeg5ThetaValue(real_idx, theta)
        case 6:
            return toDeg6ThetaValue(real_idx, theta)


def toSHBaseValue(degree, idx, phi, theta):
    real_idx = abs(idx)

    sh_weight = toSHWeight(degree, real_idx)
    assert sh_weight is not None

    sh_common_value = toSHCommonValue(idx, phi, theta)

    sh_res_value = toSHResValue(degree, real_idx, theta)

    base_value = sh_weight * sh_common_value * sh_res_value

    return base_value


def toSHBaseValues(
    degree_max: int, phis: torch.Tensor, thetas: torch.Tensor
) -> torch.Tensor:
    base_values_list = []

    for degree in range(degree_max + 1):
        for idx in range(-degree, degree + 1, 1):
            base_values_list.append(toSHBaseValue(degree, idx, phis, thetas))

    base_values = torch.vstack(base_values_list)

    return base_values
