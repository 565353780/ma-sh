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
    params: torch.Tensor,
    base_values: torch.Tensor,
    phi_idxs: torch.Tensor,
) -> torch.Tensor:
    repeat_params = params[phi_idxs]

    values_matrix = repeat_params * base_values.transpose(1, 0)

    values = torch.sum(values_matrix, dim=1)

    return values
