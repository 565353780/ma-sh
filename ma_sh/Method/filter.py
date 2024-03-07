import torch

def toMaskBoundaryMaxThetas(mask_boundary_thetas: torch.Tensor, mask_boundary_phi_idxs: torch.Tensor) -> torch.Tensor:
    max_thetas_list = []

    for i in range(mask_boundary_phi_idxs.shape[0] - 1):
        current_max_phi = torch.max(mask_boundary_thetas[mask_boundary_phi_idxs[i]:mask_boundary_phi_idxs[i + 1]])
        max_thetas_list.append(current_max_phi)

    max_thetas = torch.hstack(max_thetas_list)
    return max_thetas
