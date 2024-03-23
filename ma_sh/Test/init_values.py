import torch


def initValues(mask_params, sh_params, rotate_vectors, positions, init_mode):
    anchor_num = mask_params.shape[0]
    dtype = mask_params.dtype
    device = mask_params.device

    if init_mode == 0:
        for i in range(anchor_num):
            mask_params.data[i, 0] = i + 10.0

        for i in range(anchor_num):
            sh_params.data[i, 0] = i + 1.0

        for i in range(anchor_num):
            rotate_vectors.data[i, 0] = i

        for i in range(anchor_num):
            positions.data[i, 0] = i
    elif init_mode == 1:
        mask_params.data = (
            torch.randn(mask_params.shape, dtype=dtype).to(device) * 1000.0
        )
        sh_params.data = torch.randn(sh_params.shape, dtype=dtype).to(device) * 1000.0
        rotate_vectors.data = (
            torch.randn(rotate_vectors.shape, dtype=dtype).to(device) * 1000.0
        )
        positions.data = torch.randn(positions.shape, dtype=dtype).to(device) * 1000.0
    elif init_mode == 2:
        mask_params.data = torch.zeros_like(mask_params)
        mask_params.data[:, 0] = 1.0
        sh_params.data = torch.zeros_like(sh_params)
        sh_params.data[:, 0] = 10.0
        rotate_vectors.data = torch.zeros_like(rotate_vectors)
        positions.data = torch.zeros_like(positions)
        for i in range(anchor_num):
            positions.data[i, 2] = 15.0 * i
    return True
