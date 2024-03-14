import torch


def toMaskBaseValues(phis: torch.Tensor, degree_max: int) -> torch.Tensor:
    base_values_list = [torch.ones_like(phis)]

    for degree in range(1, degree_max + 1):
        current_phis = degree * phis
        base_values_list.append(torch.cos(current_phis))
        base_values_list.append(torch.sin(current_phis))

    base_values = torch.vstack(base_values_list)

    return base_values
