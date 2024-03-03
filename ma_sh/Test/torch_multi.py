import torch
from tqdm import trange

from ma_sh.Method.idx import toStartIdxs
from ma_sh.Method.mask import getSH2DValues, getSH2DModelValue
from ma_sh.Method.mask import getSH2DValues2, getSH2DModelValue2


def test():
    anchor_num = 40
    sh2d_degree = 5
    params = torch.randn([anchor_num, sh2d_degree * 2 + 1]).type(torch.float32)

    phi_sample_nums = torch.tensor([100, 50, 80, 40]).type(torch.int)

    phi_idxs = toStartIdxs(phi_sample_nums)

    phis = torch.randn(phi_idxs[-1]).type(torch.float32)

    print("speed getSH2DValues:")
    for _ in trange(10000):
        base_values = getSH2DValues(sh2d_degree, phis)

    getSH2DValuesC = torch.compile(getSH2DValues)
    base_values = getSH2DValuesC(sh2d_degree, phis)
    print("speed getSH2DValuesC:")
    for _ in trange(10000):
        base_values = getSH2DValuesC(sh2d_degree, phis)

    print("speed getSH2DValues2:")
    for _ in trange(10000):
        base_values = getSH2DValues2(sh2d_degree, phis)

    getSH2DValues2C = torch.compile(getSH2DValues2)
    base_values = getSH2DValues2C(sh2d_degree, phis)
    print("speed getSH2DValues2C:")
    for _ in trange(10000):
        base_values = getSH2DValues2C(sh2d_degree, phis)

    base_values = getSH2DValues(sh2d_degree, phis)

    print("speed getSH2DModelValue:")
    for _ in trange(10000):
        sh2d_values = getSH2DModelValue(phi_idxs, params, base_values)

    getSH2DModelValueC = torch.compile(getSH2DModelValue)
    sh2d_values = getSH2DModelValueC(phi_idxs, params, base_values)
    print("speed getSH2DModelValueC:")
    for _ in trange(10000):
        sh2d_values = getSH2DModelValueC(phi_idxs, params, base_values)

    print("speed getSH2DModelValue2:")
    for _ in trange(10000):
        sh2d_values = getSH2DModelValue2(phi_idxs, params, base_values)

    getSH2DModelValue2C = torch.compile(getSH2DModelValue2)
    sh2d_values = getSH2DModelValue2C(phi_idxs, params, base_values)
    print("speed getSH2DModelValue2C:")
    for _ in trange(10000):
        sh2d_values = getSH2DModelValue2C(phi_idxs, params, base_values)

    return True
