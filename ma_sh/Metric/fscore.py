import torch


@torch.no_grad()
def toFScore(
    dists2_1: torch.Tensor,
    dists2_2: torch.Tensor,
    threshold: float=0.001,
) -> torch.Tensor:
    precision_1 = torch.mean((dists2_1 < threshold).float(), dim=1)
    precision_2 = torch.mean((dists2_2 < threshold).float(), dim=1)

    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    fscore[torch.isnan(fscore)] = 0

    return fscore
