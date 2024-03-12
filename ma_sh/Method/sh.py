import torch

from ma_sh.Config.weights import W0, W1, W2, W3, W4, W5, W6


def toSHWeight(degree: int, real_idx: int) -> float:
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
        case _:
            return 0.0


def toSHCommonValue(phis: torch.Tensor, thetas: torch.Tensor, idx: int) -> torch.Tensor:
    if idx == 0:
        return torch.tensor(1.0)

    sin_thetas = torch.sin(thetas)

    pow_sin_thetas = torch.pow(sin_thetas, abs(idx))

    if idx > 0:
        cos_phis = torch.cos(1.0 * idx * phis)

        common_value = cos_phis * pow_sin_thetas

        return common_value

    sin_phis = torch.sin(-1.0 * idx * phis)

    common_value = sin_phis * pow_sin_thetas

    return common_value


def toDeg1thetasValue(thetas: torch.Tensor, real_idx: int) -> torch.Tensor:
    match real_idx:
        case 0:
            return torch.cos(thetas)
        case _:
            return torch.tensor(1.0).type(thetas.dtype).to(thetas.device)


def toDeg2thetasValue(thetas: torch.Tensor, real_idx: int) -> torch.Tensor:
    match real_idx:
        case 0:
            ct = torch.cos(thetas)
            return 3.0 * ct * ct - 1.0
        case 1:
            return torch.cos(thetas)
        case _:
            return torch.tensor(1.0).type(thetas.dtype).to(thetas.device)


def toDeg3thetasValue(thetas: torch.Tensor, real_idx: int) -> torch.Tensor:
    match real_idx:
        case 0:
            ct = torch.cos(thetas)
            return (5.0 * ct * ct - 3.0) * ct
        case 1:
            ct = torch.cos(thetas)
            return 5.0 * ct * ct - 1.0
        case 2:
            return torch.cos(thetas)
        case _:
            return torch.tensor(1.0).type(thetas.dtype).to(thetas.device)


def toDeg4thetasValue(thetas: torch.Tensor, real_idx: int) -> torch.Tensor:
    match real_idx:
        case 0:
            ct = torch.cos(thetas)
            return (35.0 * ct * ct - 30.0) * ct * ct + 3.0
        case 1:
            ct = torch.cos(thetas)
            return (7.0 * ct * ct - 3.0) * ct
        case 2:
            ct = torch.cos(thetas)
            return 7.0 * ct * ct - 1.0
        case 3:
            return torch.cos(thetas)
        case _:
            return torch.tensor(1.0).type(thetas.dtype).to(thetas.device)


def toDeg5thetasValue(thetas: torch.Tensor, real_idx: int) -> torch.Tensor:
    match real_idx:
        case 0:
            ct = torch.cos(thetas)
            return ((63.0 * ct * ct - 70.0) * ct * ct + 15.0) * ct
        case 1:
            ct = torch.cos(thetas)
            return (21.0 * ct * ct - 14.0) * ct * ct + 1.0
        case 2:
            ct = torch.cos(thetas)
            return (3.0 * ct * ct - 1.0) * ct
        case 3:
            ct = torch.cos(thetas)
            return 9.0 * ct * ct - 1.0
        case 4:
            return torch.cos(thetas)
        case _:
            return torch.tensor(1.0).type(thetas.dtype).to(thetas.device)


def toDeg6thetasValue(thetas: torch.Tensor, real_idx: int) -> torch.Tensor:
    match real_idx:
        case 0:
            ct = torch.cos(thetas)
            return ((231.0 * ct * ct - 315.0) * ct * ct + 105.0) * ct * ct - 5.0
        case 1:
            ct = torch.cos(thetas)
            return ((33.0 * ct * ct - 30.0) * ct * ct + 5.0) * ct
        case 2:
            ct = torch.cos(thetas)
            return (33.0 * ct * ct - 18.0) * ct * ct + 1.0
        case 3:
            ct = torch.cos(thetas)
            return (11.0 * ct * ct - 3.0) * ct
        case 4:
            ct = torch.cos(thetas)
            return 11.0 * ct * ct - 1.0
        case 5:
            return torch.cos(thetas)
        case _:
            return torch.tensor(1.0).type(thetas.dtype).to(thetas.device)


def toSHResValue(thetas: torch.Tensor, degree: int, real_idx: int) -> torch.Tensor:
    match degree:
        case 0:
            return torch.ones_like(thetas)
        case 1:
            return toDeg1thetasValue(thetas, real_idx)
        case 2:
            return toDeg2thetasValue(thetas, real_idx)
        case 3:
            return toDeg3thetasValue(thetas, real_idx)
        case 4:
            return toDeg4thetasValue(thetas, real_idx)
        case 5:
            return toDeg5thetasValue(thetas, real_idx)
        case 6:
            return toDeg6thetasValue(thetas, real_idx)
        case _:
            return torch.tensor(0.0).type(thetas.dtype).to(thetas.device)


def toSHBaseValue(
    phis: torch.Tensor, thetas: torch.Tensor, degree: int, idx: int
) -> torch.Tensor:
    real_idx = abs(idx)

    sh_weight = toSHWeight(degree, real_idx)
    assert sh_weight is not None

    sh_common_value = toSHCommonValue(phis, thetas, idx)

    sh_res_value = toSHResValue(thetas, degree, real_idx)

    base_value = sh_weight * sh_common_value * sh_res_value

    return base_value


def toSHBaseValues(
    phis: torch.Tensor, thetas: torch.Tensor, degree_max: int
) -> torch.Tensor:
    base_values_list = []

    for degree in range(degree_max + 1):
        for idx in range(-degree, degree + 1, 1):
            base_values_list.append(toSHBaseValue(phis, thetas, degree, idx))

    base_values = torch.vstack(base_values_list)

    return base_values
