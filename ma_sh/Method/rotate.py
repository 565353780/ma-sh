import torch

def toRotateMatrixs(rotate_vectors: torch.Tensor) -> torch.Tensor:
    thetas = torch.norm(rotate_vectors, p=2, dim=1)

    valid_theta_mask = thetas > 0.0

    divide_thetas = torch.ones([rotate_vectors.shape[0]], dtype=rotate_vectors.dtype).to(rotate_vectors.device)

    divide_thetas[valid_theta_mask] = thetas[valid_theta_mask]

    v_divide_thetas = divide_thetas.reshape(-1, 1)

    normed_rotate_vectors = rotate_vectors / v_divide_thetas

    theta_hats = torch.zeros([rotate_vectors.shape[0], 3, 3], dtype=rotate_vectors.dtype).to(rotate_vectors.device)

    theta_hats[:, 0, 1] = -1.0 * normed_rotate_vectors[:, 2]
    theta_hats[:, 0, 2] = normed_rotate_vectors[:, 1]
    theta_hats[:, 1, 0] = normed_rotate_vectors[:, 2]
    theta_hats[:, 1, 2] = -1.0 * normed_rotate_vectors[:, 0]
    theta_hats[:, 2, 0] = -1.0 * normed_rotate_vectors[:, 1]
    theta_hats[:, 2, 1] = normed_rotate_vectors[:, 0]

    identity_matrix = torch.eye(3, dtype=rotate_vectors.dtype).to(rotate_vectors.device)

    identity_matrixs = identity_matrix.repeat(rotate_vectors.shape[0], 1, 1)

    vv_thetas = thetas.reshape(-1, 1, 1)

    cos_vv_thetas = torch.cos(vv_thetas)
    sin_vv_thetas = torch.sin(vv_thetas)

    v_normed_rotate_vectors = normed_rotate_vectors.reshape(-1, 3, 1)
    h_normed_rotate_vectors = normed_rotate_vectors.reshape(-1, 1, 3)

    n_nts = torch.matmul(v_normed_rotate_vectors, h_normed_rotate_vectors)

    rotate_matrixs = cos_vv_thetas * identity_matrixs + (1.0 - cos_vv_thetas) * n_nts + sin_vv_thetas * theta_hats

    return rotate_matrixs

def toRotateVectors(rotate_matrixs: torch.Tensor) -> torch.Tensor:
    traces_list = []

    for i in range(rotate_matrixs.shape[0]):
        current_traces = torch.trace(rotate_matrixs[i])

        traces_list.append(current_traces)

    traces = torch.hstack(traces_list)

    thetas = torch.acos((traces - 1.0) * 0.5)

    sin_thetas = torch.sin(thetas)

    valid_sin_theta_mask = sin_thetas != 0.0

    divide_sin_thetas = torch.ones([rotate_matrixs.shape[0]], dtype=rotate_matrixs.dtype).to(rotate_matrixs.device)

    divide_sin_thetas[valid_sin_theta_mask] = sin_thetas[valid_sin_theta_mask]

    vv_divide_sin_thetas = divide_sin_thetas.reshape(-1, 1, 1)

    rights = 0.25 * (rotate_matrixs - rotate_matrixs.permute(0, 2, 1)) / vv_divide_sin_thetas

    normed_rotate_vectors = torch.zeros([rotate_matrixs.shape[0], 3], dtype=rotate_matrixs.dtype).to(rotate_matrixs.device)

    normed_rotate_vectors[:, 0] = rights[:, 2, 1] - rights[:, 1, 2]
    normed_rotate_vectors[:, 1] = rights[:, 0, 2] - rights[:, 2, 0]
    normed_rotate_vectors[:, 2] = rights[:, 1, 0] - rights[:, 0, 1]

    v_thetas = thetas.reshape(-1, 1)

    rotate_vectors = normed_rotate_vectors * v_thetas

    return rotate_vectors
