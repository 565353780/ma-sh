import torch


def toValues(
    params: torch.Tensor,
    base_values: torch.Tensor,
    phi_idxs: torch.Tensor,
) -> torch.Tensor:
    repeat_params = params[phi_idxs]

    values_matrix = repeat_params * base_values.transpose(1, 0)

    values = torch.sum(values_matrix, dim=1)

    return values
