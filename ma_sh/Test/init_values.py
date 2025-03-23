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
        multi_value = 1.0

        mask_params.data = (
            torch.randn(mask_params.shape, dtype=dtype).to(device) * multi_value
        )
        sh_params.data = torch.randn(sh_params.shape, dtype=dtype).to(device) * multi_value
        rotate_vectors.data = (
            torch.randn(rotate_vectors.shape, dtype=dtype).to(device) * multi_value
        )
        positions.data = torch.randn(positions.shape, dtype=dtype).to(device) * multi_value

    elif init_mode == 2:
        mask_params.data = torch.zeros_like(mask_params)
        mask_params.data[:, 0] = 1.0
        sh_params.data = torch.zeros_like(sh_params)
        for i in range(anchor_num):
            sh_params.data[:, 0] = 10.0 * i
            sh_params.data[i, 1] = 10.0 * i
        rotate_vectors.data = torch.zeros_like(rotate_vectors)
        for i in range(anchor_num):
            rotate_vectors.data[i, 0] = 1.0 * i
        positions.data = torch.zeros_like(positions)
        for i in range(anchor_num):
            positions.data[i, 2] = 15.0 * i

    elif init_mode == 3:
        mask_params.data = torch.zeros_like(mask_params)
        mask_params.data[:, 0] = 1.0
        sh_params.data = torch.zeros_like(sh_params)
        for i in range(anchor_num):
            sh_params.data[:, 0] = 10.0
            sh_params.data[i, 1] = 10.0
        rotate_vectors.data = torch.zeros_like(rotate_vectors)
        for i in range(anchor_num):
            rotate_vectors.data[i, 0] = 1.0
        positions.data = torch.zeros_like(positions)

        unit_length = int(pow(anchor_num, 1.0 / 3.0))
        unit_length = max(unit_length, 1)
        xy_num = unit_length * unit_length

        scale_weights = [30, 20, 15]

        for i in range(anchor_num):
            height_idx = int(i / xy_num)
            rel_idx = i - xy_num * height_idx
            row_idx = int(rel_idx / unit_length)
            col_idx = rel_idx - unit_length * row_idx
            positions.data[i, 0] = scale_weights[0] * row_idx
            positions.data[i, 1] = scale_weights[1] * col_idx
            positions.data[i, 2] = scale_weights[2] * height_idx

    return True
