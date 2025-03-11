import torch


@torch.no_grad()
def toNormalInConsistency(
    normals_1: torch.Tensor,
    normals_2: torch.Tensor,
    pair_idxs: torch.Tensor,
) -> torch.Tensor:
    idx_1 = pair_idxs[..., 0]
    idx_2 = pair_idxs[..., 1]

    vecs_1 = torch.gather(normals_1, 1, idx_1.unsqueeze(-1).expand(-1, -1, 3))
    vecs_2 = torch.gather(normals_2, 1, idx_2.unsqueeze(-1).expand(-1, -1, 3))

    norm_1 = torch.norm(vecs_1, dim=-1, keepdim=True)
    norm_2 = torch.norm(vecs_2, dim=-1, keepdim=True)

    vecs_1 = torch.where(norm_1 > 0, vecs_1 / norm_1, vecs_1)
    vecs_2 = torch.where(norm_2 > 0, vecs_2 / norm_2, vecs_2)

    dot_product = (vecs_1 * vecs_2).sum(dim=-1).clamp(-1, 1)

    angles = torch.acos(dot_product)
    angles = torch.rad2deg(angles)

    reverse_angles = 180.0 - angles
    min_angles = torch.minimum(angles, reverse_angles)

    average_min_angle = min_angles.mean(dim=-1, keepdim=True)

    average_angle = average_min_angle.mean(dim=-1, keepdim=True)
    return average_angle
