import torch


def toStartIdxs(data_counts: torch.Tensor) -> torch.Tensor:
    start_idxs = data_counts.detach().clone()
    for i in range(1, start_idxs.shape[0]):
        start_idxs[i] += start_idxs[i - 1]
    return start_idxs
