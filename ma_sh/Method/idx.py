import torch


def toCounts(data_list: list) -> torch.Tensor:
    counts_list = []

    for i in range(len(data_list)):
        counts_list.append(data_list[i].shape[0])

    counts = torch.tensor(counts_list).type(data_list[0].dtype).to(data_list[0].device)

    return counts


def toBoundIdxs(data_counts: torch.Tensor) -> torch.Tensor:
    bound_idxs = (
        torch.zeros(data_counts.shape[0] + 1)
        .type(data_counts.dtype)
        .to(data_counts.device)
    )
    for i in range(1, bound_idxs.shape[0]):
        bound_idxs[i] = data_counts[i - 1] + bound_idxs[i - 1]
    return bound_idxs


def toLowerIdxsList(values: torch.Tensor, max_bounds: torch.Tensor) -> list:
    lower_idxs_list = []

    for i in range(max_bounds.shape[0]):
        current_lower_idxs = torch.where(values <= max_bounds[i])[0]

        lower_idxs_list.append(current_lower_idxs)

    return lower_idxs_list
