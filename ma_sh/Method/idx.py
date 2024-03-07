import torch

def toBoundIdxs(data_counts: torch.Tensor) -> torch.Tensor:
    bound_idxs = torch.zeros(data_counts.shape[0] + 1).type(data_counts.dtype).to(data_counts.device)
    for i in range(1, bound_idxs.shape[0]):
        bound_idxs[i] = data_counts[i - 1] + bound_idxs[i - 1]
    return bound_idxs

def toInMaskSamplePolarIdxs(sample_thetas: torch.Tensor, mask_boundary_max_thetas: torch.Tensor) -> list:
    in_mask_idxs_list = []

    for i in range(mask_boundary_max_thetas.shape[0]):
        current_in_mask_idxs = torch.where(sample_thetas <= mask_boundary_max_thetas[i])[0]

        in_mask_idxs_list.append(current_in_mask_idxs)

    return in_mask_idxs_list
