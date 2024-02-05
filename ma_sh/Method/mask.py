import torch


def getSH2DValues(degree_max: int, phis: torch.Tensor) -> torch.Tensor:
    values = [torch.ones_like(phis)]

    for degree in range(1, degree_max + 1):
        current_phis = degree * phis
        values += [torch.cos(current_phis), torch.sin(current_phis)]
    return torch.cat(values, dim=-1)


def getSH2DModelValue(
    degree_max: int, params: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    assert len(params) == 2 * degree_max + 1

    value = params[0] * values[0]
    for i in range(1, len(params)):
        value = value + params[i] * values[i]

    return value
