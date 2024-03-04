import torch
from tqdm import trange

from ma_sh.Method.idx import toStartIdxs
from ma_sh.Method.mask import getSH2DBaseValues, getSH2DValues

def test():
    anchor_num = 40
    sh2d_degree = 5
    params = torch.randn([anchor_num, sh2d_degree * 2 + 1]).type(torch.float32)

    phi_sample_nums = torch.tensor([100, 50, 80, 40]).type(torch.int)

    phi_idxs = toStartIdxs(phi_sample_nums)

    phis = torch.randn(phi_idxs[-1], requires_grad=True).type(torch.float32)

    base_values = getSH2DBaseValues(sh2d_degree, phis)
    sh2d_values = getSH2DValues(phi_idxs, params, base_values)

    print("speed getSH2DBaseValues:")
    for _ in trange(10000):
        base_values = getSH2DBaseValues(sh2d_degree, phis)

    print("speed getSH2DValues:")
    for _ in trange(10000):
        sh2d_values = getSH2DValues(phi_idxs, params, base_values)

    return True
