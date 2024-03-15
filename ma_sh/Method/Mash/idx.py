import torch


def toCounts(data_list: list) -> torch.Tensor:
    counts_list = []

    for i in range(len(data_list)):
        counts_list.append(data_list[i].shape[0])

    counts = torch.tensor(counts_list).type(data_list[0].dtype).to(data_list[0].device)

    return counts


def toIdxs(data_counts: torch.Tensor) -> torch.Tensor:
    idxs_list = []

    for i in range(data_counts.shape[0]):
        current_idxs = (
            torch.ones(int(data_counts[i]), dtype=torch.int64).to(data_counts.device)
            * i
        )

        idxs_list.append(current_idxs)

    idxs = torch.hstack(idxs_list)

    return idxs


def toDataIdxs(repeat_num: int, idx_num: int) -> torch.Tensor:
    data_idxs_matrix = torch.zeros([repeat_num, idx_num], dtype=torch.int64)

    for i in range(repeat_num):
        for j in range(idx_num):
            data_idxs_matrix[i, j] = j

    data_idxs = data_idxs_matrix.reshape(-1)

    return data_idxs


def toLowerIdxsList(values: torch.Tensor, max_bounds: torch.Tensor) -> list:
    lower_idxs_list = []

    for i in range(max_bounds.shape[0]):
        current_lower_idxs = torch.where(values <= max_bounds[i])[0]

        lower_idxs_list.append(current_lower_idxs)

    return lower_idxs_list
