import torch


def toStartIdxs(data_counts: torch.Tensor) -> torch.Tensor:
    start_idxs = torch.zeros(data_counts.shape[0] + 1).type(torch.int)
    for i in range(1, start_idxs.shape[0]):
        start_idxs[i] = data_counts[i - 1] + start_idxs[i - 1]
    return start_idxs
