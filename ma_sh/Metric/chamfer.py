import torch


@torch.no_grad()
def toChamfer(
    dists2_1: torch.Tensor,
    dists2_2: torch.Tensor,
    order: float = 1.0,
) -> torch.Tensor:
    if order == 2.0:
        dists_1 = dists2_1
        dists_2 = dists2_2
    else:
        dists_1 = torch.sqrt(dists2_1)
        dists_2 = torch.sqrt(dists2_2)

        if order != 1.0:
            dists_1 = torch.pow(dists_1, order)
            dists_2 = torch.pow(dists_2, order)

    dist_1 = torch.mean(dists_1, dim=1)
    dist_2 = torch.mean(dists_2, dim=1)

    chamfer = dist_1 + dist_2
    return chamfer
