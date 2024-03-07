import torch
from tqdm import trange

from ma_sh.Method.torch_thread import setThread
from ma_sh.Method.kernel import (
    toBoundIdxs,
    getMaskBaseValues,
    getMaskValues,
)


def testMask(sh2d_degree, params, phis, phi_idxs):
    base_values = getMaskBaseValues(sh2d_degree, phis)
    assert base_values.shape[0] == params.shape[1]
    assert base_values.shape[1] == phi_idxs[-1].item()

    mask_values = getMaskValues(phi_idxs, params, base_values)
    assert mask_values.shape[0] == phi_idxs[-1].item()
    return True

def testMaskSpeed(sh2d_degree, params, phis, phi_idxs):
    base_values = getMaskBaseValues(sh2d_degree, phis)
    mask_values = getMaskValues(phi_idxs, params, base_values)
    return mask_values

def test():
    # setThread()

    anchor_num = 40
    sh2d_degree = 5
    device = 'cpu'

    params = torch.randn([anchor_num, sh2d_degree * 2 + 1]).type(torch.float32).to(device)

    phi_sample_nums = torch.randint(100, 1000, [anchor_num]).to(device)

    phi_idxs = toBoundIdxs(phi_sample_nums)

    phis = torch.randn(phi_idxs[-1], requires_grad=True).type(torch.float32).to(device)

    testMask(sh2d_degree, params, phis, phi_idxs)

    for _ in trange(10000):
        testMaskSpeed(sh2d_degree, params, phis, phi_idxs)
    return True
