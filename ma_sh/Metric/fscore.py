import torch


@torch.no_grad()
def toFScore(
    dists2_1: torch.Tensor,
    dists2_2: torch.Tensor,
    threshold: float=0.001,
) -> torch.Tensor:
    dists_1 = torch.sqrt(dists2_1)
    dists_2 = torch.sqrt(dists2_2)

    precision_1 = torch.mean((dists_1 < threshold).float(), dim=1)
    precision_2 = torch.mean((dists_2 < threshold).float(), dim=1)

    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    fscore[torch.isnan(fscore)] = 0

    return fscore
