import torch

def toBoundIdxs(data_counts: torch.Tensor) -> torch.Tensor:
    bound_idxs = torch.zeros(data_counts.shape[0] + 1).type(data_counts.dtype).to(data_counts.device)
    for i in range(1, bound_idxs.shape[0]):
        bound_idxs[i] = data_counts[i - 1] + bound_idxs[i - 1]
    return bound_idxs

def toInMaskSamplePolarIdxsList(sample_thetas: torch.Tensor, mask_boundary_max_thetas: torch.Tensor) -> list:
    in_mask_sample_polar_idxs_list = []

    for i in range(mask_boundary_max_thetas.shape[0]):
        current_in_mask_sample_polar_idxs = torch.where(sample_thetas <= mask_boundary_max_thetas[i])[0]

        in_mask_sample_polar_idxs_list.append(current_in_mask_sample_polar_idxs)

    return in_mask_sample_polar_idxs_list

def toInMaskSamplePolarCounts(in_mask_sample_polar_idxs_list: list) -> torch.Tensor:
    in_mask_sample_polar_counts_list = []

    for i in range(len(in_mask_sample_polar_idxs_list)):
        in_mask_sample_polar_counts_list.append(in_mask_sample_polar_idxs_list[i].shape[0])

    in_mask_sample_polar_counts = torch.tensor(in_mask_sample_polar_counts_list).type(
        in_mask_sample_polar_idxs_list[0].dtype).to(
        in_mask_sample_polar_idxs_list[0].device)

    return in_mask_sample_polar_counts
