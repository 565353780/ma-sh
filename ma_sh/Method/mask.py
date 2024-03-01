import torch


def getSH2DValues(degree_max: int, phis: torch.Tensor) -> torch.Tensor:
    values = [torch.ones_like(phis)]

    for degree in range(1, degree_max + 1):
        current_phis = degree * phis
        values += [torch.cos(current_phis), torch.sin(current_phis)]

    return torch.vstack(values)


def getSH2DValues2(degree_max: int, phis: torch.Tensor) -> torch.Tensor:
    return torch.vstack(
        [torch.ones_like(phis)]
        + [
            torch.vstack([torch.cos(i * phis), torch.sin(i * phis)])
            for i in range(1, degree_max + 1)
        ]
    )


def getSH2DModelValue(
    phi_idxs: torch.Tensor, params: torch.Tensor, base_values: torch.Tensor
) -> torch.Tensor:
    values = []

    for i in range(phi_idxs.shape[0] - 1):
        crop_base_values = base_values[:, phi_idxs[i]: phi_idxs[i + 1]]
        values.append(torch.matmul(params[i], crop_base_values))

    return torch.hstack(values)


def getSH2DModelValue2(
    phi_idxs: torch.Tensor, params: torch.Tensor, base_values: torch.Tensor
) -> torch.Tensor:
    return torch.hstack(
        [
            torch.matmul(params[i], base_values[:,
                         phi_idxs[i]: phi_idxs[i + 1]])
            for i in range(phi_idxs.shape[0] - 1)
        ]
    )
