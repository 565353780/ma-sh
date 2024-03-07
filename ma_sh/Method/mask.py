import torch


def toMaskBaseValues(phis: torch.Tensor, degree_max: int) -> torch.Tensor:
    base_values_list = [torch.ones_like(phis)]

    for degree in range(1, degree_max + 1):
        current_phis = degree * phis
        base_values_list.append(torch.cos(current_phis))
        base_values_list.append(torch.sin(current_phis))

    base_values = torch.vstack(base_values_list)
    return base_values


def toMaskValues(
    params: torch.Tensor, base_values: torch.Tensor, phi_idxs: torch.Tensor
) -> torch.Tensor:
    values_list = []

    for i in range(phi_idxs.shape[0] - 1):
        crop_base_values = base_values[:, phi_idxs[i]: phi_idxs[i + 1]]
        crop_values = torch.matmul(params[i], crop_base_values)
        values_list.append(crop_values)

    values = torch.hstack(values_list)
    return values
