import torch


@torch.compile()
def toBoundIdxs(data_counts: torch.Tensor) -> torch.Tensor:
    bound_idxs = torch.zeros(data_counts.shape[0] + 1).type(data_counts.dtype).to(data_counts.device)
    for i in range(1, bound_idxs.shape[0]):
        bound_idxs[i] = data_counts[i - 1] + bound_idxs[i - 1]
    return bound_idxs
