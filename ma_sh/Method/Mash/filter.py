import torch


def toMaxValues(data: torch.Tensor, data_idxs: torch.Tensor) -> torch.Tensor:
    max_values_list = []

    unique_idxs = torch.unique(data_idxs)

    for i in range(unique_idxs.shape[0]):
        current_idx = unique_idxs[i]

        current_max_value = torch.max(data[data_idxs == current_idx])

        max_values_list.append(current_max_value)

    max_values = torch.hstack(max_values_list)

    return max_values
