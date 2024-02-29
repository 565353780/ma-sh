import torch

from ma_sh.Method.idx import toStartIdxs
from ma_sh.Method.mask import getSH2DValues, getSH2DModelValue


def test():
    anchor_num = 4
    sh2d_degree = 5
    params = torch.randn([anchor_num, sh2d_degree * 2 + 1]).type(torch.float32)

    phi_sample_nums = torch.tensor([10, 5, 8, 4]).type(torch.int)

    phi_idxs = toStartIdxs(phi_sample_nums)
    print(phi_idxs)
    exit()

    phis = torch.randn(torch.sum(phi_sample_nums)).type(torch.float32)

    sh2d_values = getSH2DValues(sh2d_degree, phis)

    sh2d_dists = getSH2DModelValue(phi_sample_nums, params, sh2d_values)

    print("phis:", phis.shape)
    print("sh2d_values:", sh2d_values.shape)
    print("sh2d_dists:", sh2d_dists.shape)

    return True
