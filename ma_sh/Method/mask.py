import torch


@torch.compile()
def getSH2DBaseValues(degree_max: int, phis: torch.Tensor) -> torch.Tensor:
    base_values = [torch.ones_like(phis)]

    for degree in range(1, degree_max + 1):
        current_phis = degree * phis
        base_values += [torch.cos(current_phis), torch.sin(current_phis)]

    return torch.vstack(base_values)


# FIXME: why compile will be slow?
# @torch.compile()
def getSH2DValues(
    phi_idxs: torch.Tensor, params: torch.Tensor, base_values: torch.Tensor
) -> torch.Tensor:
    values = []

    for i in range(phi_idxs.shape[0] - 1):
        crop_base_values = base_values[:, phi_idxs[i]: phi_idxs[i + 1]]
        values.append(torch.matmul(params[i], crop_base_values))

    return torch.hstack(values)
