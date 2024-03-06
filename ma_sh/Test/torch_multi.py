import torch
from tqdm import trange

from ma_sh.Method.kernel import (
    toBoundIdxs,
    getMaskBaseValues,
    getMaskValues,
)


def merge(sh2d_degree, params, phis, phi_idxs):
    base_values = getMaskBaseValues(sh2d_degree, phis)
    mask_values = getMaskValues(phi_idxs, params, base_values)
    return True

def test():
    anchor_num = 40
    sh2d_degree = 5
    #FIXME: why cpu is more faster than gpu?
    device = 'cpu'

    params = torch.randn([anchor_num, sh2d_degree * 2 + 1]).type(torch.float32).to(device)

    phi_sample_nums = torch.tensor([1000, 500, 800, 400]).type(torch.int).to(device)

    phi_idxs = toBoundIdxs(phi_sample_nums)

    phis = torch.randn(phi_idxs[-1], requires_grad=True).type(torch.float32).to(device)

    base_values = getMaskBaseValues(sh2d_degree, phis)
    mask_values = getMaskValues(phi_idxs, params, base_values)

    merge(sh2d_degree, params, phis, phi_idxs)

    for _ in trange(10000):
        merge(sh2d_degree, params, phis, phi_idxs)

    return True
