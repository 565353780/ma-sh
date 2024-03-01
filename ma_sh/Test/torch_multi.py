import torch

from ma_sh.Method.idx import toStartIdxs
from ma_sh.Method.mask import getSH2DValues, getSH2DModelValue


def test():
    anchor_num = 4
    sh2d_degree = 5
    params = torch.randn([anchor_num, sh2d_degree * 2 + 1]).type(torch.float32)

    phi_sample_nums = torch.tensor([10, 5, 8, 4]).type(torch.int)

    phi_idxs = toStartIdxs(phi_sample_nums)

    phis = torch.randn(phi_idxs[-1]).type(torch.float32)
    print("phis:", phis.shape)

    base_values = getSH2DValues(sh2d_degree, phis)
    print("base_values:", base_values.shape)

    sh2d_values = getSH2DModelValue(phi_idxs, params, base_values)
    print("sh2d_values:", sh2d_values.shape)

    return True
