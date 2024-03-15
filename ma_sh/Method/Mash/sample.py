import torch

from ma_sh.Config.constant import PI_2, PHI_WEIGHT


def toUniformSamplePhis(sample_num: int) -> torch.Tensor:
    phis_list = [PHI_WEIGHT * (i + 0.5) for i in range(sample_num)]

    phis = torch.tensor(phis_list)

    return phis


def toUniformSampleThetas(sample_num: int) -> torch.Tensor:
    cos_thetas_list = []
    for i in range(sample_num):
        cos_thetas_list.append(1.0 - (2.0 * i + 1.0) / sample_num)

    cos_thetas = torch.tensor(cos_thetas_list)

    thetas = torch.acos(cos_thetas)

    return thetas


def toMaskBoundaryPhis(anchor_num: int, mask_boundary_sample_num: int) -> torch.Tensor:
    mask_boundary_phi_matrix = torch.zeros(anchor_num, mask_boundary_sample_num)

    for i in range(mask_boundary_sample_num):
        mask_boundary_phi_matrix[:, i] = PI_2 * i / mask_boundary_sample_num

    mask_boundary_phis = mask_boundary_phi_matrix.reshape(-1)

    return mask_boundary_phis


def toFPSIdxs(points: torch.Tensor, fps_point_num: int) -> torch.Tensor:
    """
    Input:
        points: pointcloud data, [B, N, C]
        fps_point_num: number of samples
    Return:
        fps_point_idxs: sampled pointcloud index, [B, fps_point_num]
    """
    device = points.device
    B, N, C = points.shape

    centroids = torch.zeros(B, fps_point_num, dtype=torch.long).to(device)

    distance = torch.ones(B, N).to(device) * 1e10

    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)

    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(fps_point_num):
        centroids[:, i] = farthest
        centroid = points[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((points - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return torch.sort(centroids)[0]
